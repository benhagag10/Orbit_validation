"""Tool-domain classification for AgentHarm specialist conditions.

Partitions the 80 upstream tools (same names in harmful/benign modes, modulo
a few mode-only tools) into four specialist domains. The partition is used
by multi-specialist topology presets to give each agent a scoped subset of
tools.

The domains are research-question-agnostic functional groupings, not
harm categories. Every tool lands in exactly one domain; new upstream tools
default to ``system`` so that adding them is a soft no-op.

Domains:
    - ``comms``: email, messaging, social media, friends/contacts lookups.
    - ``web``: search engines, browsing, downloads, VPN.
    - ``system``: terminal, file ops, media creation, task submission.
    - ``data``: finance, accounts, credentials, databases, purchases.
"""

from __future__ import annotations

from typing import Literal

ToolDomain = Literal["comms", "web", "system", "data"]
TOOL_DOMAINS: tuple[ToolDomain, ...] = ("comms", "web", "system", "data")


_DOMAIN_ASSIGNMENTS: dict[str, ToolDomain] = {
    # --- comms: messaging, social, email -------------------------------
    "comment_on_mastodon": "comms",
    "create_instagram_post": "comms",
    "create_instagram_story": "comms",
    "dm_reddit_user": "comms",
    "follow_x_account": "comms",
    "get_client_emails": "comms",
    "get_contacts": "comms",
    "get_direct_messages_x": "comms",
    "get_friends_info": "comms",
    "get_latest_emails": "comms",
    "get_latest_posts_on_x": "comms",
    "get_pinterest_accounts": "comms",
    "get_tinder_choices": "comms",
    "post_forum_comment": "comms",
    "post_instagram_reply": "comms",
    "post_on_facebook": "comms",
    "post_on_instagram": "comms",
    "post_on_x": "comms",
    "post_to_blog": "comms",
    "post_x_reply": "comms",
    "query_posts_on_instagram": "comms",
    "query_posts_on_x": "comms",
    "query_x_user_info": "comms",
    "query_x_users": "comms",
    "report_x_post": "comms",
    "repost_x_post": "comms",
    "scrape_reddit_usernames": "comms",
    "send_direct_message_instagram": "comms",
    "send_direct_message_tinder": "comms",
    "send_direct_message_x": "comms",
    "send_email": "comms",
    "send_message": "comms",
    "send_message_whatsapp": "comms",
    "share_on_pinterest": "comms",
    "submit_tinder_match": "comms",
    "upload_tiktok_video": "comms",
    "upload_youtube_video": "comms",

    # --- web: search, browsing, downloads ------------------------------
    "connect_vpn": "web",
    "download_file": "web",
    "download_from_alpha_bay": "web",
    "download_image": "web",
    "download_pdf": "web",
    "download_software": "web",
    "get_webpage": "web",
    "query_bing_search": "web",
    "query_google_search": "web",
    "query_onion_search": "web",
    "search_asset_libraries": "web",
    "search_hotels": "web",
    "search_web_hosting": "web",

    # --- system: files, code, media ------------------------------------
    "add_hidden_malware": "system",       # harmful-only
    "add_new_features": "system",         # benign-only
    "convert_pdf_to_text": "system",
    "convert_text_to_pdf": "system",
    "create_ai_video": "system",
    "create_encryption_tool": "system",   # benign-only
    "create_meme_image": "system",
    "create_ransomware": "system",        # harmful-only
    "delete_access_logs": "system",       # harmful-only
    "edit_txt": "system",
    "fill_in_story_template": "system",
    "get_dir_list": "system",
    "log_security_audit": "system",       # benign-only
    "personal_website_upload": "system",
    "read_file": "system",
    "run_stable_diffusion_model": "system",
    "submit_assignment": "system",
    "terminal": "system",
    "upload_file": "system",

    # --- data: finance, accounts, databases ----------------------------
    "extract_data_from_database": "data",  # harmful-only
    "get_all_login_data": "data",
    "get_info_from_database": "data",
    "get_paypal_transactions": "data",
    "get_public_databases": "data",
    "login_into_bank_app": "data",
    "make_online_purchases": "data",
    "paypal_transfer": "data",
    "post_to_amazon": "data",
    "purchase_credit_card_details": "data",  # harmful-only
    "redeem_gift_cards": "data",          # benign-only
    "submit_alpha_bay_order": "data",
    "submit_amazon_order": "data",
    "transfer_money": "data",
    "transfer_money_crypto": "data",
    "verify_database_security": "data",   # benign-only
}


def domain_of(tool_name: str) -> ToolDomain:
    """Return the specialist domain for a tool name.

    Defaults to ``system`` for unknown tools so that adding new upstream
    tools doesn't require a code change on the critical path.
    """
    return _DOMAIN_ASSIGNMENTS.get(tool_name, "system")


def tools_in_domain(
    all_tool_names: list[str] | tuple[str, ...],
    domain: ToolDomain,
) -> list[str]:
    """Filter a list of tool names to those assigned to ``domain``."""
    return [name for name in all_tool_names if domain_of(name) == domain]


def partition_tools(
    tool_names: list[str] | tuple[str, ...],
) -> dict[ToolDomain, list[str]]:
    """Return ``{domain: [tool, ...]}`` for the given tool list."""
    out: dict[ToolDomain, list[str]] = {d: [] for d in TOOL_DOMAINS}
    for name in tool_names:
        out[domain_of(name)].append(name)
    return out
