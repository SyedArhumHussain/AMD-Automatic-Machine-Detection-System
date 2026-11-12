#!/usr/bin/env python3
# elastic_dashboard.py — AMD Detection Dashboard (v10.3 - HEAVY LOAD HARDENED)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
import base64
import numpy as np
import time
import warnings
import requests
import re
import json
import os
from urllib.parse import quote
warnings.filterwarnings('ignore')

ELASTICSEARCH_URL = ""
ELASTIC_API_KEY = ""
INDEX_PREFIX = ""
AUDIO_USER = ""
AUDIO_PASSWORD = ""
_ENCODED_PASSWORD = quote(AUDIO_PASSWORD, safe="")
AUDIO_URL_PREFIX = f"https://{AUDIO_USER}:{_ENCODED_PASSWORD}@{AUDIO_USER}.your-storagebox.de/audio_storage"

# Fixed hard cap for robust heavy-data handling
HARD_LIMIT = 60_000


class ElasticsearchDashboard:
    def __init__(self):
        self.client = self._create_client()
        self.index_pattern = f"{INDEX_PREFIX}-*"
        self.audio_session = requests.Session()
        self.audio_session.auth = (AUDIO_USER, AUDIO_PASSWORD)
        self.audio_session.headers.update({'User-Agent': 'Mozilla/5.0'})

    def _process_api_key(self):
        try:
            decoded = base64.b64decode(ELASTIC_API_KEY).decode('utf-8')
            if ':' in decoded:
                return tuple(decoded.split(':', 1))
            return ELASTIC_API_KEY
        except:
            return ELASTIC_API_KEY

    def _create_client(self):
        try:
            client = Elasticsearch(hosts=[ELASTICSEARCH_URL], api_key=self._process_api_key(),
                                   request_timeout=60, verify_certs=False,
                                   max_retries=3, retry_on_timeout=True)
            return client if client.ping() else None
        except Exception as e:
            st.error(f"Connection error: {e}")
            return None

    def fix_audio_url(self, url):
        if not url or not isinstance(url, str) or len(url.strip()) < 10:
            return url
        url = url.strip()
        match = re.search(r'/([A-Za-z0-9_\-]+?)(?:_\d{8}_\d{6})?\.(mp3|wav)$', url)
        if match:
            return f"{AUDIO_URL_PREFIX}/{match.group(1)}.{match.group(2).lower()}"
        return url

    def fetch_audio(self, url):
        try:
            if not url or len(url.strip()) < 10:
                return None
            r = self.audio_session.get(url, timeout=15)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                alt = url[:-4] + ('.wav' if url.endswith('.mp3') else '.mp3')
                ar = self.audio_session.get(alt, timeout=15)
                if ar.status_code == 200:
                    return ar.content
            return None
        except Exception:
            return None

    def get_server_ip_counts(self, start_date=None, end_date=None, size=200):
        try:
            must = []
            if start_date or end_date:
                tr = {}
                if start_date: tr['gte'] = int(start_date.timestamp() * 1000)
                if end_date:   tr['lte'] = int(end_date.timestamp() * 1000)
                must.append({"range": {"timestamp": tr}})
            body = {
                "size": 0,
                "query": {"bool": {"must": must or [{"match_all": {}}]}},
                "aggs": {"server_ips": {"terms": {"field": "server_ip.keyword", "size": size,
                    "order": {"_count": "desc"}, "missing": "(unknown)"}}},
            }
            resp = self.client.search(index=self.index_pattern, body=body, request_timeout=30)
            return [{"server_ip": b["key"], "count": int(b["doc_count"])}
                    for b in resp.get("aggregations", {}).get("server_ips", {}).get("buckets", [])]
        except Exception:
            return []

    def query_data(self, start_date=None, end_date=None, customer_number=None,
                   decision=None, server_ip=None, progress_callback=None):
        must = []
        if start_date or end_date:
            tr = {}
            if start_date: tr['gte'] = int(start_date.timestamp() * 1000)
            if end_date:   tr['lte'] = int(end_date.timestamp() * 1000)
            must.append({"range": {"timestamp": tr}})
        if customer_number and customer_number.strip():
            must.append({"match_phrase": {"customer_number": customer_number}})
        if decision and decision != "All":
            must.append({"match_phrase": {"decision": decision}})
        if server_ip and server_ip != "All" and str(server_ip).strip():
            must.append({"match_phrase": {"server_ip": server_ip}})

        query_body = {
            "query": {"bool": {"must": must or [{"match_all": {}}]}},
            "sort": [{"timestamp": {"order": "desc"}}],
        }
        try:
            data = self._fetch_records(query_body, HARD_LIMIT, progress_callback)
            if not data:
                return pd.DataFrame()
            return self._postprocess_dataframe(pd.DataFrame(data))
        except Exception as e:
            st.error(f"Query error: {e}")
            return pd.DataFrame()

    def _fetch_records(self, query_body, limit, progress_callback=None):
        # Reduced batch size + shorter TTL = less ES memory pressure, faster scroll release
        BATCH_SIZE = 2_000
        SCROLL_TTL = "1m"
        results = []
        scroll_id = None
        try:
            resp = self.client.search(index=self.index_pattern,
                body={"size": BATCH_SIZE, **query_body}, scroll=SCROLL_TTL, request_timeout=90)
            scroll_id = resp.get("_scroll_id")
            total = resp.get("hits", {}).get("total", {})
            if isinstance(total, dict): total = total.get("value", limit)
            total = min(int(total or limit), limit)
            hits = resp.get("hits", {}).get("hits", [])
            results.extend(h["_source"] for h in hits)
            if progress_callback: progress_callback(len(results), total)

            while hits and len(results) < limit:
                try:
                    resp = self.client.scroll(scroll_id=scroll_id, scroll=SCROLL_TTL, request_timeout=90)
                except Exception as scroll_err:
                    # Scroll expired or network issue — return what we have
                    st.warning(f"Scroll interrupted at {len(results):,} records: {scroll_err}")
                    break
                scroll_id = resp.get("_scroll_id")
                hits = resp.get("hits", {}).get("hits", [])
                if not hits: break
                remaining = limit - len(results)
                if len(hits) > remaining: hits = hits[:remaining]
                results.extend(h["_source"] for h in hits)
                if progress_callback: progress_callback(len(results), total)
            return results
        finally:
            if scroll_id:
                try: self.client.clear_scroll(scroll_id=scroll_id)
                except: pass

    def _postprocess_dataframe(self, df):
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        if 'audio_start_time_epoch_ms' in df.columns:
            df['audio_start_datetime'] = pd.to_datetime(df['audio_start_time_epoch_ms'], unit='ms', errors='coerce')

        if 'audio_url' in df.columns:
            if 'vid_sanitized' in df.columns and 'audio_format' in df.columns:
                vs = df['vid_sanitized'].fillna('').astype(str).str.strip()
                fmt = df['audio_format'].fillna('').astype(str).str.strip()
                rebuilt = AUDIO_URL_PREFIX + "/" + vs + "." + fmt
                mask = (vs != '') & (fmt != '')
                # Only run fix_audio_url on rows that DON'T have the rebuild path (saves huge CPU on 60K)
                original = df['audio_url'].fillna('').astype(str)
                df['audio_url'] = np.where(mask, rebuilt, original)
                # Apply regex fix only to non-rebuilt rows
                non_mask = ~mask
                if non_mask.any():
                    df.loc[non_mask, 'audio_url'] = df.loc[non_mask, 'audio_url'].map(self.fix_audio_url)
            else:
                df['audio_url'] = df['audio_url'].fillna('').astype(str).map(self.fix_audio_url)
            df['has_audio'] = df['audio_url'].astype(str).str.len() > 10
        else:
            df['has_audio'] = False
            df['audio_url'] = ''

        if 'vid' not in df.columns: df['vid'] = ''
        if 'server_ip' not in df.columns: df['server_ip'] = ''
        else: df['server_ip'] = df['server_ip'].fillna('').astype(str)
        return df

    def get_stats_from_data(self, df):
        if df.empty: return None
        n = len(df)
        hm = df['decision'].str.upper() == 'HUMAN' if 'decision' in df.columns else pd.Series(False, index=df.index)
        mm = df['decision'].str.upper() == 'MACHINE' if 'decision' in df.columns else pd.Series(False, index=df.index)
        return {
            'total_count': n,
            'human_count': int(hm.sum()), 'machine_count': int(mm.sum()),
            'avg_confidence': df['confidence'].mean() if 'confidence' in df.columns else 0,
            'avg_response_time': df['response_time_ms'].mean() if 'response_time_ms' in df.columns else 0,
            'human_avg_response_time': df.loc[hm, 'response_time_ms'].mean() if 'response_time_ms' in df.columns else 0,
            'machine_avg_response_time': df.loc[mm, 'response_time_ms'].mean() if 'response_time_ms' in df.columns else 0,
        }


st.set_page_config(page_title="AMD Detection Dashboard", page_icon="📊", layout="wide")
st.title("AMD Detection Dashboard")
st.markdown("### Real-time Human vs Machine Detection Analytics")

c1, c2, c3 = st.columns([1, 6, 1])
with c3:
    if st.button("Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

@st.cache_resource
def init_es(): return ElasticsearchDashboard()
es_dashboard = init_es()

if es_dashboard.client:
    st.sidebar.header("Filters")
    st.sidebar.subheader("Time Range")
    # Hours 1-12
    time_options = [f"Last {h} Hour{'s' if h > 1 else ''}" for h in range(1, 13)] + ["Custom Range"]
    time_option = st.sidebar.selectbox("Select Time Range", options=time_options, index=0)

    # IMPORTANT: compute 'now' fresh each run so Refresh gives the correct window
    now = datetime.now()
    if time_option == "Custom Range":
        cc1, cc2 = st.sidebar.columns(2)
        with cc1:
            sdi = st.date_input("Start Date", value=now.date() - timedelta(days=1))
            sti = st.time_input("Start Time", value=now.time())
        with cc2:
            edi = st.date_input("End Date", value=now.date())
            eti = st.time_input("End Time", value=now.time())
        start_date, end_date = datetime.combine(sdi, sti), datetime.combine(edi, eti)
        hrs = (end_date - start_date).total_seconds() / 3600
    else:
        hrs = int(time_option.split()[1])
        start_date, end_date = now - timedelta(hours=hrs), now

    st.sidebar.write(f"Window: {hrs:g}h ({start_date.strftime('%m-%d %H:%M')} → {end_date.strftime('%m-%d %H:%M')})")

    st.sidebar.subheader("Detection Type")
    decision_filter = st.sidebar.selectbox("Show results for:", ["All", "HUMAN", "MACHINE"])
    st.sidebar.subheader("Customer Filter")
    customer_filter = st.sidebar.text_input("Customer Number", placeholder="Enter customer number...")

    st.sidebar.subheader("Server IP Filter")
    _ips = es_dashboard.get_server_ip_counts(start_date, end_date, size=200)
    server_ip_filter = st.sidebar.selectbox("Dialer Server IP",
        options=["All"] + [b["server_ip"] for b in _ips], index=0)

    with st.spinner(f"Fetching records ({start_date.strftime('%H:%M')} → {end_date.strftime('%H:%M')})..."):
        _pb = st.empty(); _pt = st.empty()
        def _cb(f, t):
            _pb.progress(min(f / HARD_LIMIT, 1.0))
            _pt.caption(f"Fetched {f:,} records...")
        t0 = time.time()
        df_data = es_dashboard.query_data(
            start_date=start_date, end_date=end_date,
            customer_number=customer_filter or None,
            decision=decision_filter if decision_filter != "All" else None,
            server_ip=server_ip_filter if server_ip_filter != "All" else None,
            progress_callback=_cb)
        _pb.empty(); _pt.empty()

        if not df_data.empty:
            total_records = len(df_data)
            st.caption(f"✓ Loaded {total_records:,} records in {time.time()-t0:.1f}s  |  Window: {hrs:g}h")

            # Show warning for very heavy loads
            if total_records >= HARD_LIMIT:
                st.info(f"ℹ️ Showing latest {HARD_LIMIT:,} records (hard cap). Narrow your time window for full coverage.")

            stats = es_dashboard.get_stats_from_data(df_data)

            st.subheader("Key Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1: st.metric("Total Records", f"{stats['total_count']:,}")
            with m2: st.metric("Human Detections", f"{stats['human_count']:,}",
                delta=f"{stats['human_count']/stats['total_count']*100:.1f}%" if stats['total_count'] else "0%")
            with m3: st.metric("Machine Detections", f"{stats['machine_count']:,}",
                delta=f"{stats['machine_count']/stats['total_count']*100:.1f}%" if stats['total_count'] else "0%")
            with m4: st.metric("Average Confidence", f"{stats['avg_confidence']:.2%}")
            with m5: st.metric("Average Response Time", f"{stats['avg_response_time']:.0f} ms")

            st.subheader("Detection Analytics")
            dc1, dc2 = st.columns(2)
            with dc1:
                fig1 = go.Figure(data=[go.Pie(labels=['Human','Machine'],
                    values=[stats['human_count'],stats['machine_count']],
                    marker=dict(colors=['#2ecc71','#e74c3c']), hole=0.4,
                    textinfo='label+percent', pull=[0.05,0])])
                fig1.update_layout(title="Human vs Machine Distribution", height=450)
                st.plotly_chart(fig1, use_container_width=True)
            with dc2:
                fig2 = go.Figure(data=[go.Pie(labels=['Human RT','Machine RT'],
                    values=[stats['human_avg_response_time'],stats['machine_avg_response_time']],
                    marker=dict(colors=['#2ecc71','#e74c3c']), hole=0.4,
                    textinfo='label+percent', pull=[0.05,0])])
                fig2.update_layout(title="Average Response Time Comparison", height=450)
                st.plotly_chart(fig2, use_container_width=True)

            # ============================================
            # SERVER IPs (NO AMD Customer Name column)
            # ============================================
            st.subheader("Server IP's")
            if 'server_ip' in df_data.columns:
                ip_series = df_data['server_ip'].fillna('').astype(str).replace('', '(unknown)')
                ip_counts_df = ip_series.value_counts().reset_index()
                ip_counts_df.columns = ["Client IP's", "Count"]
            else:
                ip_counts_df = pd.DataFrame(columns=["Client IP's", "Count"])

            if not ip_counts_df.empty:
                total_ips = len(ip_counts_df)
                total_calls = int(ip_counts_df["Count"].sum())

                NAMES_FILE = "server_ip_names.json"
                def load_ip_names():
                    if os.path.exists(NAMES_FILE):
                        with open(NAMES_FILE) as f: return json.load(f)
                    return {}
                def save_ip_names(d):
                    with open(NAMES_FILE, "w") as f: json.dump(d, f, indent=2)

                ip_names = load_ip_names()
                editor_df = ip_counts_df.copy()
                editor_df["Customer Name"] = editor_df["Client IP's"].map(ip_names).fillna("")

                st.markdown(f"**{total_ips} unique server IP(s) · {total_calls:,} total calls**")

                edited_df = st.data_editor(
                    editor_df,
                    column_config={
                        "Client IP's": st.column_config.TextColumn("Client IP's", disabled=True),
                        "Count": st.column_config.NumberColumn("Count", disabled=True, format="%d"),
                        "Customer Name": st.column_config.TextColumn("Customer Name",
                            help="Assign a friendly name to this server IP"),
                    },
                    hide_index=True, use_container_width=True, key="server_ip_editor",
                    column_order=["Client IP's", "Count", "Customer Name"],
                )

                col_save, col_dl = st.columns([1, 3])
                with col_save:
                    if st.button("💾 Save Names", type="primary", use_container_width=True):
                        new_map = {}
                        for _, r in edited_df.iterrows():
                            ip = r["Client IP's"]
                            nm = r["Customer Name"].strip() if pd.notna(r["Customer Name"]) else ""
                            if nm: new_map[ip] = nm
                        save_ip_names(new_map)
                        st.success("Saved!")
                        st.rerun()
                with col_dl:
                    st.download_button(f"📥 Download Server IP Breakdown ({total_ips} IPs)",
                        data=ip_counts_df.to_csv(index=False),
                        file_name=f"server_ip_counts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv", use_container_width=True)

                top15 = ip_counts_df.head(15).iloc[::-1]
                fig_sip = go.Figure(data=[go.Bar(x=top15["Count"], y=top15["Client IP's"],
                    orientation='h', marker=dict(color='#3498db'),
                    text=top15["Count"].apply(lambda v: f"{int(v):,}"), textposition='outside')])
                fig_sip.update_layout(title=f"Top {min(15, total_ips)} Server IPs by Volume",
                    height=520, plot_bgcolor='#fafafa', showlegend=False)
                st.plotly_chart(fig_sip, use_container_width=True)
            else:
                st.info("No server_ip data available.")

            # ============================================
            # Time series — adaptive resample so 60K doesn't blow up Plotly
            # ============================================
            st.subheader("Call Volume Over Time")
            if 'audio_start_datetime' in df_data.columns:
                dft = df_data[['audio_start_datetime']].dropna()
                if not dft.empty:
                    # Adaptive bucket size based on time window and record count
                    if hrs <= 2:
                        rule = '1min'
                    elif hrs <= 6:
                        rule = '5min'
                    elif hrs <= 12:
                        rule = '15min'
                    else:
                        rule = 'H'
                    dft2 = dft.set_index('audio_start_datetime').resample(rule).size().reset_index()
                    dft2.columns = ['audio_start_datetime','count']
                    # Cap chart points to avoid plotly lag
                    if len(dft2) > 500:
                        dft2 = dft2.iloc[::max(1, len(dft2)//500)]
                    fig = px.line(dft2, x='audio_start_datetime', y='count',
                        title=f"Number of Calls Over Time (bucket: {rule})", line_shape='spline')
                    fig.update_traces(line=dict(color='#3498db', width=3))
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            # ============================================
            # Human Detection Responses — page-slice FIRST, never iterate full set
            # ============================================
            st.subheader("Human Detection Responses")
            if 'decision' in df_data.columns:
                human_mask = df_data['decision'].str.upper() == 'HUMAN'
                ht = int(human_mask.sum())
                if ht > 0:
                    st.markdown(f"**Total Human Detections: {ht}**")
                    ipp = 10; tp = (ht + ipp - 1) // ipp
                    if tp > 1:
                        if 'human_page' not in st.session_state: st.session_state.human_page = 1
                        # Guard against stale page number after filter change
                        if st.session_state.human_page > tp:
                            st.session_state.human_page = 1
                        _, pc, _ = st.columns([2,1,2])
                        with pc:
                            sp = st.selectbox("Page", list(range(1, tp+1)),
                                index=st.session_state.human_page-1, key="hp_sel")
                            if sp != st.session_state.human_page:
                                st.session_state.human_page = sp; st.rerun()
                    cp = st.session_state.get('human_page', 1) if tp > 1 else 1
                    si, ei = (cp-1)*ipp, min(cp*ipp, ht)

                    # Slice first, then materialize — never iterate 60K
                    human_slice = df_data.loc[human_mask].iloc[si:ei]

                    for idx, row in human_slice.iterrows():
                        parts = [f"Customer: {row.get('customer_number','?')}"]
                        if pd.notna(row.get('server_ip')) and str(row.get('server_ip','')).strip():
                            parts.append(f"SrvIP: {row['server_ip']}")
                        if pd.notna(row.get('datetime')):
                            parts.append(f"Date: {row['datetime'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row['datetime'], pd.Timestamp) else row['datetime']}")
                        with st.expander(" - ".join(parts)):
                            st.json({
                                "customer_number": row.get('customer_number','N/A'),
                                "server_ip": row.get('server_ip','N/A'),
                                "customer_name": row.get('customer_name','N/A'),
                                "decision": "HUMAN",
                                "confidence": f"{row.get('confidence',0):.2%}",
                                "transcript": (row.get('transcript','') or '')[:500],
                                "audio_url": row.get('audio_url',''),
                            })
                            aurl = str(row.get('audio_url', '')).strip()
                            if aurl and len(aurl) > 10:
                                st.markdown("**🔊 Audio Playback**")
                                if st.button("▶️ Load & Play Audio", key=f"audio_{idx}_{cp}"):
                                    with st.spinner("Loading audio..."):
                                        ad = es_dashboard.fetch_audio(aurl)
                                        if ad: st.audio(ad, format='audio/mp3')
                                        else: st.error("Failed to load audio.")
                                st.markdown(f"[⬇ Download Audio]({aurl})", unsafe_allow_html=True)
                else:
                    st.info("No HUMAN detection responses found")

            # ============================================
            # Detailed Records — 10 per page, SLICE BEFORE transforming (critical for 60K)
            # ============================================
            st.subheader("Detailed Records")
            st.markdown(f"**Total Records: {total_records:,}**")

            fields = ['customer_number', 'server_ip', 'customer_name',
                      'audio_start_time_str', 'decision', 'transcript', 'audio_url']
            avail = [f for f in fields if f in df_data.columns]
            if avail:
                # Pagination first
                ipp2 = 10; tp2 = (total_records + ipp2 - 1) // ipp2
                if tp2 > 1:
                    if 'detailed_page' not in st.session_state: st.session_state.detailed_page = 1
                    # Guard against stale page after filter change
                    if st.session_state.detailed_page > tp2:
                        st.session_state.detailed_page = 1
                    _, dpc, _ = st.columns([2,1,2])
                    with dpc:
                        dp = st.selectbox("Page", list(range(1, tp2+1)),
                            index=st.session_state.detailed_page-1, key="dp_sel")
                        if dp != st.session_state.detailed_page:
                            st.session_state.detailed_page = dp; st.rerun()
                cp2 = st.session_state.get('detailed_page', 1) if tp2 > 1 else 1
                si2, ei2 = (cp2-1)*ipp2, min(cp2*ipp2, total_records)

                # CRITICAL FIX: slice BEFORE transforming. This means the expensive
                # .apply() runs on 10 rows, not 60,000.
                paginated_raw = df_data[avail].iloc[si2:ei2].copy()

                rn = {'customer_number':'Customer Number', 'server_ip':'Server IP',
                      'customer_name':'Customer Name', 'audio_start_time_str':'Audio Start Time',
                      'decision':'Decision', 'transcript':'Transcript', 'audio_url':'🔊 Audio'}
                paginated = paginated_raw.rename(columns={k:v for k,v in rn.items() if k in paginated_raw.columns})

                if 'Transcript' in paginated.columns:
                    paginated['Transcript'] = paginated['Transcript'].apply(
                        lambda x: x[:100]+"..." if isinstance(x, str) and len(x) > 100 else x)
                if '🔊 Audio' in paginated.columns:
                    paginated['🔊 Audio'] = paginated['🔊 Audio'].apply(
                        lambda x: f'<a href="{x}" target="_blank" style="text-decoration:none; display:inline-block; background:#3498db; color:white; padding:4px 8px; border-radius:4px; font-size:12px;">▶️ Play</a>' if pd.notna(x) and len(str(x).strip()) > 10 else '<span style="color:#999;">—</span>')

                # Build HTML table — only 10 rows now, safe
                hdr = ''.join(f'<th style="background:#2c3e50;color:#ecf0f1;padding:10px 12px;font-size:12px;text-align:left;white-space:nowrap;border-bottom:2px solid #34495e;">{c}</th>' for c in paginated.columns)
                rows = ''
                for i, (_, row) in enumerate(paginated.iterrows()):
                    dec = str(row.get('Decision','')).upper()
                    bg = '#e8f5e9' if dec == 'HUMAN' else ('#f8f9fa' if i%2==0 else '#fff')
                    cells = ''
                    for col in paginated.columns:
                        v = row[col]
                        s = 'padding:9px 12px;font-size:13px;border-bottom:1px solid #eee;'
                        if col == 'Decision':
                            if dec == 'HUMAN': s += 'font-weight:600;color:#27ae60;'
                            elif dec == 'MACHINE': s += 'font-weight:600;color:#c0392b;'
                        elif col in ('Customer Number','Server IP','Customer Name'):
                            s += 'font-family:monospace;font-size:12px;'
                        elif col == 'Transcript':
                            s += 'max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
                        elif col == 'Audio Start Time':
                            s += 'font-family:monospace;font-size:11px;white-space:nowrap;'
                        cells += f'<td style="{s}">{v if pd.notna(v) else ""}</td>'
                    rows += f'<tr style="background:{bg};">{cells}</tr>'

                st.markdown(f'''
                <div style="overflow-x:auto;border-radius:8px;border:1px solid #ddd;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
                    <table style="width:100%;border-collapse:collapse;">
                    <thead>{hdr}</thead><tbody>{rows}</tbody>
                    </table>
                </div>
                ''', unsafe_allow_html=True)

                col_a, col_b, col_c = st.columns([2, 2, 2])
                with col_a:
                    st.markdown(f"**Showing {si2+1}–{ei2} of {total_records:,} records · Page {cp2}/{tp2}**")
                with col_b:
                    st.download_button(f"📄 Download Current Page ({len(paginated)})",
                        data=paginated.to_csv(index=False),
                        file_name=f"amd_page_{cp2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv", use_container_width=True)
                with col_c:
                    if total_records > 25_000:
                        with st.expander(f"📥 Download All {total_records:,} (large)"):
                            if st.button("Generate CSV", key="gen_csv", use_container_width=True):
                                with st.spinner("Building CSV..."):
                                    # Only export the visible columns, not the internal helper columns
                                    export_cols = [c for c in avail if c in df_data.columns]
                                    fc = df_data[export_cols].to_csv(index=False)
                                st.download_button(f"⬇ Download {total_records:,}", data=fc,
                                    file_name=f"amd_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv", use_container_width=True, key="full_dl")
                    else:
                        export_cols = [c for c in avail if c in df_data.columns]
                        st.download_button(f"📥 Download All {total_records:,} Records",
                            data=df_data[export_cols].to_csv(index=False),
                            file_name=f"amd_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv", use_container_width=True)
        else:
            st.warning("No records found. Try adjusting your search criteria.")
else:
    st.error("Failed to connect to Elasticsearch.")