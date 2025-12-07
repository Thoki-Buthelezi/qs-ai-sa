# app.py  — Cloud-ready, page-splitting, smart-skip mode (Option 1 + C)
import streamlit as st
import anthropic
import pandas as pd
import json
import io
import base64
import time
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF

# ---------------------------
# Page & Secrets
# ---------------------------
st.set_page_config(page_title="AI Quantity Surveyor | Phase 0", layout="wide")

# Load API key from Streamlit secrets
try:
    API_KEY = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    st.error("🔒 Add your Anthropic API key to Streamlit Secrets first (see README or .streamlit/secrets.toml)")
    st.stop()

client = anthropic.Anthropic(api_key=API_KEY)

# ---------------------------
# Limits & Prompts
# ---------------------------
MAX_FILE_MB = 30
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

# Prompt: Smart-skip behavior embedded
PAGE_PROMPT_SINGLE = """
You are an expert South African quantity surveyor following SANS 1200 standards.
You will receive ONE PDF page (single page). Inspect the page and do the following:

1) If the page DOES NOT contain any measurable building elements (e.g., title page, revision index, blank page, general notes), then respond with an empty JSON array exactly like this:
[]

2) Otherwise, EXTRACT ALL measurable quantities visible on this page (walls, slabs, beams, doors, windows, roofing, plumbing, electrical items, finishes, etc.). For each item produce an object with:
- item_description (string)
- unit (string, e.g., "m²", "m³", "each", "lm")
- quantity (number) — if you cannot determine a numeric quantity, set quantity to null
- notes (optional string)

3) Output ONLY a JSON array (no explanation, no extra text). Example:
[
  {"item_description":"Brickwork in 230mm wall","unit":"m²","quantity":120.5,"notes":"face brick, openings excluded"},
  {"item_description":"Concrete slab on ground","unit":"m³","quantity":12.0,"notes":"25MPa, 150mm"}
]

BE CONSERVATIVE: Do NOT invent dimensions or items not visible on the page. If ambiguous, put explanatory text in notes.
DO NOT HALLCUINATE items not there.
"""

# Short page-type helper prompt (optional)
PAGE_TYPE_PROMPT = """
Based on the single page provided, classify it as one of:
- Floor Plan
- Section/Elevation
- MEP
- Schedule
- Unknown

Return exactly one of the above words/phrases and nothing else.
"""

# ---------------------------
# Utilities
# ---------------------------
def encode_pdf_to_base64(pdf_bytes: bytes) -> str:
    return base64.b64encode(pdf_bytes).decode("utf-8")

def extract_json_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract first JSON array from text. Return [] if none."""
    if not text or not isinstance(text, str):
        return []
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1 or end < start:
        return []
    candidate = text[start:end+1]
    try:
        data = json.loads(candidate)
        if isinstance(data, list):
            return data
    except Exception:
        import re
        candidate2 = re.sub(r'```(?:json)?\s*', '', candidate)
        candidate2 = re.sub(r'```\s*', '', candidate2)
        candidate2 = re.sub(r',\s*([}\]])', r'\1', candidate2)
        try:
            data = json.loads(candidate2)
            if isinstance(data, list):
                return data
        except Exception:
            return []
    return []

# Minimal classification helper
def classify_item_category(description: str) -> str:
    desc = (description or "").lower()
    if any(k in desc for k in ["slab", "beam", "column", "foundation", "concrete", "pile"]):
        return "Structural"
    if any(k in desc for k in ["wall", "partition", "door", "window", "brick", "cladding", "finish"]):
        return "Architectural"
    if any(k in desc for k in ["roof", "truss", "roofing", "gutter"]):
        return "Roofing"
    if any(k in desc for k in ["pipe", "duct", "plumbing", "electrical", "wire", "switch", "cable"]):
        return "MEP"
    if any(k in desc for k in ["paint", "tile", "flooring", "skirting"]):
        return "Finishes"
    return "Other"

def normalize_item(it: Dict[str, Any], source_file: str, page_number: int, page_type: str) -> Dict[str, Any]:
    desc = it.get("item_description") or it.get("description") or ""
    unit = it.get("unit") or ""
    qty_raw = it.get("quantity") if "quantity" in it else it.get("qty")
    notes = it.get("notes") or ""
    qty = None
    if qty_raw is not None:
        try:
            if isinstance(qty_raw, str):
                qty = float(qty_raw.replace(",", "").strip())
            else:
                qty = float(qty_raw)
        except Exception:
            qty = None
            notes = (notes + " | quantity parse failed").strip(" |")
    return {
        "source_file": source_file,
        "page_number": page_number,
        "page_type": page_type,
        "category": classify_item_category(desc),
        "item_description": desc,
        "unit": unit,
        "quantity": qty,
        "notes": notes
    }

# ---------------------------
# Cost system (user CSV + fallback)
# ---------------------------
DEFAULT_RATES = {
    "brickwork": {"unit_cost": 850.0, "unit": "m²", "source": "Demo default"},
    "concrete": {"unit_cost": 1450.0, "unit": "m³", "source": "Demo default"},
    "roofing": {"unit_cost": 700.0, "unit": "m²", "source": "Demo default"},
    "door": {"unit_cost": 1200.0, "unit": "each", "source": "Demo default"},
    "window": {"unit_cost": 950.0, "unit": "each", "source": "Demo default"},
    "plumbing": {"unit_cost": 300.0, "unit": "each", "source": "Demo default"},
    "electrical": {"unit_cost": 250.0, "unit": "each", "source": "Demo default"},
    "flooring": {"unit_cost": 400.0, "unit": "m²", "source": "Demo default"},
    "paint": {"unit_cost": 180.0, "unit": "m²", "source": "Demo default"},
    "default": {"unit_cost": 500.0, "unit": "", "source": "Demo default"}
}

def load_price_csv(uploaded_csv) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded_csv)
        df.columns = [c.strip().lower() for c in df.columns]
        if "keyword" not in df.columns or "unit_cost" not in df.columns:
            st.warning("Price CSV must contain 'keyword' and 'unit_cost' columns. Falling back to defaults if necessary.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.warning(f"Could not read price CSV: {e}. Using defaults.")
        return pd.DataFrame()

def match_unit_cost_from_table(desc: str, price_table: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if price_table is None or price_table.empty:
        return None
    desc_l = (desc or "").lower()
    for _, row in price_table.iterrows():
        kw = str(row.get("keyword", "")).lower()
        if not kw:
            continue
        if kw in desc_l:
            try:
                unit_cost = float(row["unit_cost"])
            except:
                unit_cost = None
            unit = row.get("unit", "") if "unit" in row.index else ""
            notes = row.get("cost_notes", "") if "cost_notes" in row.index else "User price list"
            return {"unit_cost": unit_cost, "unit": unit or "", "cost_notes": notes}
    return None

def apply_costs(items: List[Dict[str, Any]], price_table: pd.DataFrame, allow_defaults: bool=True) -> List[Dict[str, Any]]:
    for it in items:
        desc = it["item_description"] or ""
        matched = match_unit_cost_from_table(desc, price_table) if (price_table is not None and not price_table.empty) else None
        if matched and matched.get("unit_cost") is not None:
            it["unit_cost"] = matched["unit_cost"]
            it["cost_unit"] = matched.get("unit", "")
            it["cost_notes"] = matched.get("cost_notes", "User price match")
        else:
            # fallback to keyword-based defaults
            assigned = None
            for k, v in DEFAULT_RATES.items():
                if k != "default" and k in desc.lower():
                    assigned = v
                    break
            if assigned is None:
                assigned = DEFAULT_RATES["default"]
            if allow_defaults:
                it["unit_cost"] = assigned["unit_cost"]
                it["cost_unit"] = assigned.get("unit", "")
                it["cost_notes"] = f"Fallback default ({assigned.get('source')})"
            else:
                it["unit_cost"] = None
                it["cost_unit"] = ""
                it["cost_notes"] = "No pricing applied"
        if it.get("quantity") is not None and it.get("unit_cost") is not None:
            try:
                it["total_cost"] = float(it["quantity"]) * float(it["unit_cost"])
            except:
                it["total_cost"] = None
        else:
            it["total_cost"] = None
    return items

# ---------------------------
# Anthropic call w/ fallback & rate-safe retry
# ---------------------------
def call_anthropic_page(prompt_text: str, pdf_page_b64: str, model_candidates: List[str], max_attempts:int=3) -> str:
    """
    Sends a single-page document to Anthropic using provided prompt_text.
    Retries on transient errors and backs off on rate limits.
    """
    messages = [{"role":"user", "content":[{"type":"text","text":prompt_text},
                                           {"type":"document","source":{"type":"base64","media_type":"application/pdf","data":pdf_page_b64}}]}]
    last_err = None
    for model in model_candidates:
        attempt = 1
        while attempt <= max_attempts:
            try:
                resp = client.messages.create(model=model, max_tokens=4000, temperature=0.0, messages=messages)
                raw_text = None
                if hasattr(resp, "content") and isinstance(resp.content, list) and len(resp.content) > 0:
                    first = resp.content[0]
                    if isinstance(first, dict):
                        raw_text = first.get("text")
                    else:
                        raw_text = getattr(first, "text", None)
                if raw_text is None:
                    try:
                        raw_text = resp.get("completion") or resp.get("text") or str(resp)
                    except:
                        raw_text = str(resp)
                return raw_text or ""
            except Exception as e:
                last_err = e
                # If rate limit, backoff longer
                err_s = str(e).lower()
                if "rate_limit" in err_s or "429" in err_s:
                    backoff = 2 ** attempt
                    time.sleep(backoff)
                else:
                    time.sleep(0.6)
                attempt += 1
                continue
    # if all failed
    raise last_err

# ---------------------------
# PDF splitting helper (PyMuPDF)
# ---------------------------
def split_pdf_to_pages_bytes(pdf_bytes: bytes) -> List[bytes]:
    """
    Return a list of PDF bytes, each a single-page PDF extracted from the input.
    Uses PyMuPDF.
    """
    out_pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for pno in range(doc.page_count):
        single = fitz.open()  # new empty PDF
        single.insert_pdf(doc, from_page=pno, to_page=pno)
        page_bytes = single.write()  # returns bytes
        out_pages.append(page_bytes)
        single.close()
    doc.close()
    return out_pages

# ---------------------------
# UI & Main Flow
# ---------------------------
st.title("🛠️AI Quantity Surveyor | Phase 0")
st.markdown(
    "Upload multi-page PDF drawings. The app splits each PDF into pages, sends one page per AI call (smart-skip if page has no measurable items), merges extracted BOQ items, and optionally applies your pricing CSV."
)

st.sidebar.header("Settings & Pricing")
price_csv = st.sidebar.file_uploader("Upload optional price CSV (keyword, unit_cost, unit, cost_notes)", type=["csv"])
allow_defaults = st.sidebar.checkbox("Allow fallback demo defaults if no match", value=True)
model_choice = st.sidebar.selectbox("Preferred Claude model", ["claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219"])
st.sidebar.markdown("Model fallback will be used automatically if preferred model is unavailable.")

# Load price table if provided
if price_csv:
    price_table = load_price_csv(price_csv)
else:
    price_table = pd.DataFrame()

uploaded_files = st.file_uploader("Drop multi-page PDF drawings here (one or more)", type=["pdf"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload PDFs to begin. Tip: for best results upload clean, high-resolution architectural/MEP drawings.")
    st.stop()

# Validate file sizes
oversize = [f.name for f in uploaded_files if f.size > MAX_FILE_BYTES]
if oversize:
    st.error(f"The following files exceed the {MAX_FILE_MB}MB limit: {', '.join(oversize)}. Reduce file size and try again.")
    st.stop()

if st.button("🟢 Extract BOQ (page-by-page)"):
    model_candidates = [model_choice, "claude-3-7-sonnet-20250219", "claude-sonnet-4-20250514"]
    aggregated_items: List[Dict[str,Any]] = []
    raw_per_file: Dict[str, List[str]] = {}
    progress = st.progress(0)
    total_pages_est = 0
    # First compute total pages for progress
    for f in uploaded_files:
        try:
            doc = fitz.open(stream=f.read(), filetype="pdf")
            total_pages_est += doc.page_count
            doc.close()
        except Exception:
            total_pages_est += 1
    processed_pages = 0

    # Process each file
    for fidx, uploaded in enumerate(uploaded_files, start=1):
        uploaded.seek(0)
        fname = uploaded.name
        st.write(f"Processing **{fname}** ({fidx}/{len(uploaded_files)}) ...")
        pdf_bytes = uploaded.read()
        try:
            pages_bytes = split_pdf_to_pages_bytes(pdf_bytes)
        except Exception as e:
            st.error(f"Failed to split PDF {fname}: {e}")
            continue

        raw_per_file[fname] = []
        # process each page (one AI call per page)
        for pno, page_bytes in enumerate(pages_bytes, start=1):
            processed_pages += 1
            progress.progress(int(processed_pages/total_pages_est*100))
            # prepare base64 page
            page_b64 = encode_pdf_to_base64(page_bytes)

            # Compose prompt (single page extraction + smart skip)
            prompt = PAGE_PROMPT_SINGLE

            try:
                raw_text = call_anthropic_page(prompt, page_b64, model_candidates=model_candidates, max_attempts=3)
                raw_per_file[fname].append(raw_text or "")
                # Parse JSON array from this page
                page_items = extract_json_from_text(raw_text or "")
                # If no items, skip (smart-skip)
                if not page_items:
                    # nothing found on this page
                    continue
                # Normalize each item with metadata
                for it in page_items:
                    norm = normalize_item(it, fname, pno, "Unknown")
                    # Optionally run a tiny page-type detection per page (cheap): we skip to save tokens
                    aggregated_items.append(norm)
            except Exception as e:
                st.warning(f"AI extraction for {fname} page {pno} failed: {e}")
                raw_per_file[fname].append(f"ERROR: {e}")
                # continue to next page

    progress.progress(100)
    st.success("Extraction complete (page-by-page). Merging results...")

    # Apply pricing
    aggregated_with_costs = apply_costs(aggregated_items, price_table, allow_defaults)

    # Build DataFrame
    if not aggregated_with_costs:
        st.info("No BOQ items were extracted. Inspect raw AI outputs below for debugging.")
    else:
        df = pd.DataFrame(aggregated_with_costs)
        df["quantity_missing"] = df["quantity"].isna()
        st.header("Extracted BOQ (merged)")
        st.write(f"Total items extracted: **{len(df)}**")
        if not price_table.empty:
            st.success("Prices: using uploaded price list.")
        else:
            if allow_defaults:
                st.warning("Prices: using demo fallback defaults (for demo only). Encourage customers to upload a price CSV.")
            else:
                st.info("No pricing applied (fallback defaults disabled).")

        # Show full BOQ with missing quantity highlight
        def highlight_missing(row):
            if "quantity_missing" not in row:
                 return ["" for _ in row]
            return ["background-color: yellow" if row["quantity_missing"] else "" for _ in row]

        # Guarantee column exists
        if "quantity_missing" not in df.columns:
            df["quantity_missing"] = df["quantity"].isna()
        
        styled_df = df.drop(columns=["quantity_missing"]).style.apply(
    highlight_missing, axis=1
)
        st.dataframe(styled_df, use_container_width=True)

        # Filters
        st.subheader("Filters")
        cats = ["All"] + sorted(df["category"].fillna("Other").unique().tolist())
        pages = ["All"] + sorted(df["page_type"].fillna("Unknown").unique().tolist())
        csel = st.selectbox("Category", cats)
        psel = st.selectbox("Page Type", pages)
        filtered = df.copy()
        if csel != "All":
            filtered = filtered[filtered["category"] == csel]
        if psel != "All":
            filtered = filtered[filtered["page_type"] == psel]
        st.write(f"Filtered items: **{len(filtered)}**")
        st.dataframe(filtered.drop(columns=["quantity_missing"]).style.apply(highlight_missing, axis=1), use_container_width=True)

        # Raw AI outputs expander
        with st.expander("Show raw AI outputs (per file/page)"):
            for fname, raws in raw_per_file.items():
                st.markdown(f"**{fname}**")
                for p_idx, txt in enumerate(raws, start=1):
                    st.markdown(f"- Page {p_idx}")
                    st.text_area(f"raw: {fname} page {p_idx}", value=txt or "No output", height=160)

        # Export
        st.subheader("Export")
        csv_bytes = filtered.drop(columns=["quantity_missing"]).to_csv(index=False).encode("utf-8")
        st.download_button("📥Download filtered BOQ CSV", csv_bytes, file_name="boq_filtered.csv", mime="text/csv")
        try:
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
                filtered.drop(columns=["quantity_missing"]).to_excel(writer, index=False, sheet_name="BOQ")
            towrite.seek(0)
            st.download_button("📥Download filtered BOQ Excel", towrite, file_name="boq_filtered.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.warning(f"Excel export failed: {e}")

    st.balloons()

# Footer & notes
st.markdown("---")
st.caption("Notes: This cloud-safe edition splits PDFs into pages and sends one page per AI call. Smart-skip mode returns [] for non-relevant pages. Encourage users to upload clean, scaled drawings and/or an explicit price CSV for accurate costing.")
st.caption("Build by a UJ Computer Science student. Free pilots this week. DM for tweaks.")
