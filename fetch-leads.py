#!/usr/bin/env python3
"""
Smartlead Global Leads Exporter (two-table architecture)

Outputs two CSV files:
- leads.csv: one row per lead with core attributes only (no custom_fields JSON)
- lead_custom_fields.csv: one row per custom field key/value linked by lead_id

Features:
- Uses GET https://server.smartlead.ai/api/v1/leads/global-leads with api_key, offset, limit, created_at_gt
- Paginates with offset and hasMore; filters locally by inclusive date range
- Unnests custom_fields into a narrow table with normalized keys
- Attempts numeric downcasting to int8 when possible; also exposes int, float, bool columns
- Retries with backoff and honors Retry-After for 429
- Logs total time taken and totals written

Usage example:
  python fetch-leads.py \
    --api-key YOUR_KEY \
    --start-date 2024-03-01 \
    --end-date 2024-04-30 \
    --leads-output leads.csv \
    --custom-fields-output lead_custom_fields.csv \
    --limit 100
"""

import argparse
import csv
import json
import re
import sys
import time
from datetime import date, datetime, time as dtime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover - fallback for different urllib3 layouts
    from urllib3.util import Retry  # type: ignore


API_BASE_URL = "https://server.smartlead.ai/api/v1/leads/global-leads"
USER_AGENT = "Smartlead-GlobalLeads-Exporter/3.0 (+python requests)"
API_MAX_LIMIT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Smartlead global leads into two CSVs: leads and custom_fields (flattened).",
    )
    parser.add_argument("--api-key", required=True, help="Smartlead API key (api_key query param)")
    parser.add_argument("--start-date", required=True, help="Inclusive start date YYYY-MM-DD (used for created_at_gt and local filtering)")
    parser.add_argument("--end-date", required=True, help="Inclusive end date YYYY-MM-DD (local filtering)")
    parser.add_argument("--leads-output", default="leads.csv", help="Leads CSV output path")
    parser.add_argument("--custom-fields-output", default="lead_custom_fields.csv", help="Custom fields CSV output path")
    parser.add_argument("--limit", type=int, default=API_MAX_LIMIT, help=f"Items per page (max {API_MAX_LIMIT})")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional cap on total pages fetched")
    return parser.parse_args()


def ensure_limit(limit: int) -> int:
    if limit <= 0:
        return 1
    return min(limit, API_MAX_LIMIT)


def parse_yyyy_mm_dd(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Invalid date '{value}'. Expect YYYY-MM-DD.") from exc


def to_utc_dt_start_of_day(d: date) -> datetime:
    return datetime.combine(d, dtime(0, 0, 0, 0, tzinfo=timezone.utc))


def to_utc_dt_end_of_day(d: date) -> datetime:
    return datetime.combine(d, dtime(23, 59, 59, 999999, tzinfo=timezone.utc))


def parse_api_iso8601(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except Exception:
        return None


def build_retrying_session(total_retries: int = 10, backoff_factor: float = 1.0) -> requests.Session:
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        backoff_factor=backoff_factor,
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def request_page(
    session: requests.Session,
    api_key: str,
    offset: int,
    limit: int,
    created_at_gt: str,
) -> Dict[str, Any]:
    params = {
        "api_key": api_key,
        "offset": offset,
        "limit": limit,
        "created_at_gt": created_at_gt,
    }
    resp = session.get(API_BASE_URL, params=params, timeout=60)
    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After")
        sleep_s = 2.0
        try:
            if retry_after:
                sleep_s = float(retry_after)
        except Exception:
            sleep_s = 2.0
        time.sleep(max(sleep_s, 2.0))
        resp = session.get(API_BASE_URL, params=params, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    return resp.json()


_NON_ALNUM_UNDERSCORE = re.compile(r"[^a-z0-9_]+")


def normalize_cf_key(key: str) -> str:
    """Normalize custom field key to a stable, comparable form.

    - Lowercases
    - Trims whitespace
    - Replaces spaces and punctuation with underscores
    - Collapses multiple underscores and trims edges
    """
    k = (key or "").strip().lower()
    k = k.replace(" ", "_").replace("-", "_")
    k = _NON_ALNUM_UNDERSCORE.sub("_", k)
    k = re.sub(r"_+", "_", k)
    k = k.strip("_")
    return k


def coerce_numeric_and_bool(value: Any) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[bool], str]:
    """Attempt to parse value to int8, int, float, bool.

    Returns tuple: (value_int8, value_int, value_float, value_bool, value_text)
    - Only one of int8/int/float/bool is typically non-None; value_text keeps compact string form.
    - For integers within -128..127, value_int8 is set; otherwise value_int holds larger ints.
    - Floats use dot decimal; commas are stripped.
    """
    if value is None:
        return None, None, None, None, ""

    # Already-typed primitives
    if isinstance(value, bool):
        return None, None, None, bool(value), "true" if value else "false"
    if isinstance(value, int):
        if -128 <= value <= 127:
            return int(value), None, None, None, str(value)
        return None, int(value), None, None, str(value)
    if isinstance(value, float):
        return None, None, float(value), None, ("%.10g" % value)

    s = str(value).strip()
    if s == "":
        return None, None, None, None, ""

    # Booleans (various capitalizations)
    low = s.lower()
    if low in ("true", "false"):
        return None, None, None, (low == "true"), low

    # Remove thousands separators, spaces
    num = s.replace(",", "").replace(" ", "")
    # Try int
    if re.fullmatch(r"[-+]?\d+", num or ""):
        try:
            iv = int(num)
            if -128 <= iv <= 127:
                return iv, None, None, None, str(iv)
            return None, iv, None, None, str(iv)
        except Exception:
            pass
    # Try float
    if re.fullmatch(r"[-+]?\d*\.\d+", num or "") or re.fullmatch(r"[-+]?\d+\.\d*", num or ""):
        try:
            fv = float(num)
            return None, None, fv, None, ("%.10g" % fv)
        except Exception:
            pass

    # Default: keep as short text
    return None, None, None, None, s


def write_csv_headers(
    leads_writer: csv.DictWriter,
    cf_writer: csv.DictWriter,
) -> None:
    leads_writer.writeheader()
    cf_writer.writeheader()


def main() -> None:
    args = parse_args()
    api_key = args.api_key.strip()
    if not api_key:
        raise SystemExit("API key is required.")

    start_d = parse_yyyy_mm_dd(args.start_date)
    end_d = parse_yyyy_mm_dd(args.end_date)
    if end_d < start_d:
        raise SystemExit("end-date must be on or after start-date.")

    limit = ensure_limit(args.limit)

    session = build_retrying_session()

    # Prepare CSV writers
    leads_fieldnames = [
        "id",
        "email",
        "first_name",
        "last_name",
        "company_name",
        "website",
        "company_url",
        "phone_number",
        "location",
        "linkedin_profile",
        "created_at",
        "user_id",
        # Small, informative derived columns
        "has_campaigns",
        "campaigns_count",
    ]
    cf_fieldnames = [
        "lead_id",
        "key_original",
        "key_normalized",
        "value_text",
        "value_bool",
        "value_int8",
        "value_int",
        "value_float",
    ]

    t0 = time.perf_counter()
    total_leads = 0
    total_cf_rows = 0

    start_utc = to_utc_dt_start_of_day(start_d)
    end_utc = to_utc_dt_end_of_day(end_d)
    created_at_gt_param = start_d.strftime("%Y-%m-%d")

    offset = 0
    page = 0

    with open(args.leads_output, "w", encoding="utf-8", newline="") as lf, \
         open(args.custom_fields_output, "w", encoding="utf-8", newline="") as cf:
        leads_writer = csv.DictWriter(lf, fieldnames=leads_fieldnames, extrasaction="ignore")
        cf_writer = csv.DictWriter(cf, fieldnames=cf_fieldnames, extrasaction="ignore")
        write_csv_headers(leads_writer, cf_writer)

        while True:
            if args.max_pages is not None and page >= args.max_pages:
                break

            data = request_page(
                session=session,
                api_key=api_key,
                offset=offset,
                limit=limit,
                created_at_gt=created_at_gt_param,
            )

            items = data.get("data") or []
            has_more = bool(data.get("hasMore"))
            resp_skip = data.get("skip")

            print(f"Fetched page {page+1} | offset={offset} | items={len(items)} | hasMore={has_more}", file=sys.stderr)

            for lead in items:
                created_at = parse_api_iso8601(lead.get("created_at"))
                if created_at is None:
                    continue
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                if created_at < start_utc or created_at > end_utc:
                    continue

                # Write lead row (no custom_fields JSON)
                campaigns = lead.get("campaigns") or []
                lead_row = {
                    "id": lead.get("id"),
                    "email": lead.get("email"),
                    "first_name": lead.get("first_name"),
                    "last_name": lead.get("last_name"),
                    "company_name": lead.get("company_name"),
                    "website": lead.get("website"),
                    "company_url": lead.get("company_url"),
                    "phone_number": lead.get("phone_number"),
                    "location": lead.get("location"),
                    "linkedin_profile": lead.get("linkedin_profile"),
                    "created_at": lead.get("created_at"),
                    "user_id": lead.get("user_id"),
                    "has_campaigns": 1 if isinstance(campaigns, list) and len(campaigns) > 0 else 0,
                    "campaigns_count": len(campaigns) if isinstance(campaigns, list) else 0,
                }
                # Normalize scalars to strings for CSV compactness
                for k, v in list(lead_row.items()):
                    if v is None:
                        lead_row[k] = ""
                leads_writer.writerow(lead_row)
                total_leads += 1

                # Flatten custom_fields to second table
                cf_obj = lead.get("custom_fields") or {}
                if isinstance(cf_obj, dict):
                    for key, raw_val in cf_obj.items():
                        key_original = str(key)
                        key_norm = normalize_cf_key(key_original)
                        v_int8, v_int, v_float, v_bool, v_text = coerce_numeric_and_bool(raw_val)
                        cf_row = {
                            "lead_id": lead.get("id"),
                            "key_original": key_original,
                            "key_normalized": key_norm,
                            "value_text": v_text,
                            "value_bool": ("true" if v_bool else ("false" if v_bool is not None else "")),
                            "value_int8": v_int8 if v_int8 is not None else "",
                            "value_int": v_int if v_int is not None else "",
                            "value_float": ("%.10g" % v_float) if v_float is not None else "",
                        }
                        cf_writer.writerow(cf_row)
                        total_cf_rows += 1
                else:
                    # If custom_fields is unexpectedly non-dict, preserve as a single row
                    key_original = "__root__"
                    key_norm = normalize_cf_key(key_original)
                    v_int8, v_int, v_float, v_bool, v_text = coerce_numeric_and_bool(cf_obj)
                    cf_row = {
                        "lead_id": lead.get("id"),
                        "key_original": key_original,
                        "key_normalized": key_norm,
                        "value_text": v_text,
                        "value_bool": ("true" if v_bool else ("false" if v_bool is not None else "")),
                        "value_int8": v_int8 if v_int8 is not None else "",
                        "value_int": v_int if v_int is not None else "",
                        "value_float": ("%.10g" % v_float) if v_float is not None else "",
                    }
                    cf_writer.writerow(cf_row)
                    total_cf_rows += 1

            # Advance pagination
            offset = (resp_skip if isinstance(resp_skip, int) else offset) + limit
            page += 1
            if not has_more or len(items) == 0:
                break

    t1 = time.perf_counter()
    elapsed = t1 - t0

    # Final report to stdout
    print(f"Leads written: {total_leads}")
    print(f"Custom field rows written: {total_cf_rows}")
    print(f"Leads CSV: {args.leads_output}")
    print(f"Custom Fields CSV: {args.custom_fields_output}")
    print(f"Time taken: {elapsed:.2f}s")


if __name__ == "__main__":
    main()


