"""Stripe Checkout integration for pay-per-draft.

Environment variables:
    STRIPE_SECRET_KEY      — Stripe API secret key
    STRIPE_PRICE_ID        — Price ID for the draft product
    STRIPE_WEBHOOK_SECRET  — Webhook signing secret
    FQ_BASE_URL            — Base URL for redirects (e.g. https://fantasyquant.com)

For development, set STRIPE_SECRET_KEY=dev to bypass payment.
"""

from __future__ import annotations

import os


STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "dev")
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
FQ_BASE_URL = os.environ.get("FQ_BASE_URL", "http://localhost:8000")

_DEV_MODE = STRIPE_SECRET_KEY == "dev"


def is_dev_mode() -> bool:
    return _DEV_MODE


def create_checkout_session(session_id: str) -> str:
    """Create a Stripe Checkout session and return the checkout URL.

    In dev mode, returns a direct link to the draft room (no payment).
    """
    if _DEV_MODE:
        return f"{FQ_BASE_URL}/draft/{session_id}"

    import stripe
    stripe.api_key = STRIPE_SECRET_KEY

    checkout = stripe.checkout.Session.create(
        mode="payment",
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        success_url=f"{FQ_BASE_URL}/draft/{session_id}?paid=1",
        cancel_url=f"{FQ_BASE_URL}/configure?cancelled=1",
        metadata={"session_id": session_id},
    )
    return checkout.url


def verify_webhook(payload: bytes, sig_header: str) -> dict | None:
    """Verify a Stripe webhook signature and return the event data.

    Returns None if verification fails.
    """
    if _DEV_MODE:
        return None

    import stripe
    stripe.api_key = STRIPE_SECRET_KEY

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET,
        )
        return event
    except (ValueError, stripe.error.SignatureVerificationError):
        return None
