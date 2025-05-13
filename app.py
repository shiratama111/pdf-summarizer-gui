"""
Streamlit + Gemini + Google Drive 連携版
---------------------------------------
● アップロードした PDF は自動で Drive に保存  
● Drive 上の PDF を直接選択して要約することも可能
"""

# ========= ライブラリ =========
import io
import os
import zipfile
from types import SimpleNamespace

from google import genai
from google.genai import types as gtypes

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

import streamlit as st
from dotenv import load_dotenv

# ========= 定数 =========
MODEL_ID   = "gemini-2.5-pro-exp-03-25"  # 無料枠あり版
MAX_CHARS  = 10_000                      # 追加指示の上限
SCOPES     = [
    "https://www.googleapis.com/auth/drive.file",     # 書き込み
    "https://www.googleapis.com/auth/drive.readonly", # 読み取り
]
DEFAULT_DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")  # 任意：.env にフォルダ ID を入れておく

# ========= 初期化 =========
load_dotenv()
api_key_env = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="PDF Summarizer", page_icon="📄")
st.title("📄 Gemini PDF Summarizer (Drive 連携)")

# ---------- サイドバー (API キー入力) ----------
with st.sidebar:
    st.header("🔑 API Key")
    api_key_input = st.text_input(
        "Google API Key",
        value=api_key_env if api_key_env else "",
        type="password",
        placeholder="AI Studio で発行したキーを入れてください",
    )
    st.markdown(
        "*キーはブラウザには保存されません。ダミー値で始まると失敗します。*",
        help=".env に書いておくと毎回入力不要"
    )

if not api_key_input:
    st.warning("API Key を入力するか .env に設定してください。")
    st.stop()

client = genai.Client(api_key=api_key_input)

# ========= Google Drive ヘルパ =========
@st.cache_resource(show_spinner=False)
def get_drive_service():
    """
    OAuth 認証を行い `googleapiclient.discovery.Resource` を返す。
    初回のみブラウザが開く。トークンは token.json にキャッシュ。
    """
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    else:
        if not os.path.exists("client_secret.json"):
            st.error("client_secret.json が見つかりません。Google Cloud で OAuth クライアントを発行して配置してください。")
            st.stop()
        flow  = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
        creds = flow.run_local_server(port=0, prompt="consent", authorization_prompt_message="")
        with open("token.json", "w") as f:
            f.write(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)

drive_service = get_drive_service()

def save_pdf_to_drive(filename: str, pdf_bytes: bytes, folder_id: str | None = DEFAULT_DRIVE_FOLDER_ID):
    """アップロードされた PDF を Drive に保存"""
    file_meta = {"name": filename}
    if folder_id:
        file_meta["parents"] = [folder_id]
    media = MediaIoBaseUpload(io.BytesIO(pdf_bytes), mimetype="application/pdf", resumable=False)
    drive_service.files().create(body=file_meta, media_body=media, fields="id").execute()

def list_drive_pdfs(max_files: int = 1000):
    """Drive 内の PDF を取得 (id, name, size) リストで返す"""
    res = drive_service.files().list(
        q="mimeType='application/pdf' and trashed=false",
        pageSize=max_files,
        fields="files(id,name,size)"
    ).execute()
    return res.get("files", [])

def download_drive_file(file_id: str) -> bytes:
    """Drive からファイルをダウンロードしてバイト列で返す"""
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read()

# ========= UI =========
uploaded_files = st.file_uploader(
    "PDF ファイルを選択してください（複数可）",
    type=["pdf"],
    accept_multiple_files=True,
)

# --- Drive 上の PDF を選択 ---
with st.expander("📂 Google Drive から選択する"):
    drive_pdf_list   = list_drive_pdfs()
    choices_display  = ["-- 未選択 --"] + [f"{f['name']} ({int(f['size'])/1024:.0f} KB)" for f in drive_pdf_list]
    drive_choice_str = st.selectbox("要約したい PDF を Drive から選ぶ", choices_display, index=0)
    if drive_choice_str != "-- 未選択 --":
        idx       = choices_display.index(drive_choice_str) - 1
        drive_sel = drive_pdf_list[idx]  # dict: id, name, size
        # ダウンロードして UploadedFile 風オブジェクトに変換
        drive_bytes = download_drive_file(drive_sel["id"])
        uploaded_files = list(uploaded_files)  # StreamlitUploader が tuple の場合あり
        uploaded_files.append(
            SimpleNamespace(name=drive_sel["name"], read=lambda: drive_bytes)
        )

prompt_extra = st.text_area(
    "追加指示 (オプション。10,000 文字まで)",
    placeholder="例）敬語で / 箇条書きで 5 行 / ポイントと結論を分けて など",
    max_chars=MAX_CHARS,
    key="prompt_extra",
)

# --- プログレス表示用 ---
progress_bar = st.progress(0)

# ========= メイン処理 =========
if st.button("▶ まとめて要約", disabled=len(uploaded_files) == 0):
    summaries = {}
    total     = len(uploaded_files)

    for idx, file in enumerate(uploaded_files, 1):
        try:
            pdf_bytes = file.read()
            # Drive にアップロードしたファイルは保存済みなので判定
            if not isinstance(file, SimpleNamespace):  # ローカルからのアップロード
                save_pdf_to_drive(file.name, pdf_bytes)

            parts = [
                gtypes.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                f"この PDF を日本語で 3 行に要約してください。{prompt_extra}"
            ]
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=parts
            )
            summaries[file.name] = response.text.strip()

            progress_bar.progress(idx / total, text=f"{idx}/{total} 件 完了")
        except Exception as e:
            summaries[file.name] = f"【エラー】: {e}"

    # --- 結果表示 ---
    st.success("✅ すべて完了！")
    for name, summary in summaries.items():
        st.subheader(name)
        st.text_area(
            label=f"要約 ({name})",
            value=summary,
            height=150,
            key=f"summary_{name}",
            label_visibility="collapsed"
        )

    # --- ZIP 生成 & ダウンロード ---
    with io.BytesIO() as buffer:
        with zipfile.ZipFile(buffer, "w") as zf:
            for name, summary in summaries.items():
                txt_name = name.rsplit(".", 1)[0] + "_summary.txt"
                zf.writestr(txt_name, summary.encode("utf-8"))
        buffer.seek(0)
        st.download_button(
            "⬇ すべてまとめてダウンロード（ZIP）",
            buffer,
            file_name="summaries.zip",
            mime="application/zip",
        )
