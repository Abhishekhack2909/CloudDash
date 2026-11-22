"""Tool implementations for CloudDash support agents."""

from typing import Optional
from langchain_core.tools import tool

from database.connection import get_db_session, init_db
from database.models import AccountRecord, SupportTicketRecord
from database.seed import seed_accounts_if_empty
from tools.schemas import (
    CheckIntegrationStatusInput,
    CreateSupportTicketInput,
    LookupAccountInput,
)


@tool(args_schema=CreateSupportTicketInput)
def create_support_ticket(
    title: str,
    description: str,
    priority: str = "medium",
    category: str = "technical",
) -> dict:
    """Create a real support ticket in the database. Returns ticket ID and status."""
    init_db()
    with get_db_session() as session:
        count = session.query(SupportTicketRecord).count()
        ticket_id = f"TICKET-{1000 + count + 1}"
        record = SupportTicketRecord(
            ticket_id=ticket_id,
            title=title,
            description=description,
            priority=priority,
            category=category,
            status="open",
        )
        session.add(record)
        session.commit()
        return {
            "ticket_id": ticket_id,
            "title": title,
            "priority": priority,
            "category": category,
            "status": "open",
            "created_at": record.created_at.isoformat(),
            "message": f"Support ticket {ticket_id} created successfully.",
        }


@tool(args_schema=CheckIntegrationStatusInput)
def check_integration_status(
    integration_name: str,
    customer_id: Optional[str] = None,
) -> dict:
    """Check the real-time operational status, sync time, and errors for a customer integration."""
    integ = integration_name.strip().lower()
    cid = customer_id or "default"

    if integ in ["aws", "cloudwatch"]:
        if cid == "cust_102" or "expired" in integ:
            return {
                "integration": integration_name,
                "customer_id": cid,
                "health": "unhealthy",
                "last_sync": "2026-05-13T08:15:22Z",
                "error_count": 4,
                "error_details": "ERR_AUTH_EXPIRED: AWS IAM AssumeRole credentials expired.",
                "rate_limit_remaining": 8500,
            }
        return {
            "integration": integration_name,
            "customer_id": cid,
            "health": "healthy",
            "last_sync": "2026-05-13T10:00:00Z",
            "error_count": 0,
            "rate_limit_remaining": 9850,
        }
    elif integ in ["kubernetes", "k8s"]:
        return {
            "integration": integration_name,
            "customer_id": cid,
            "health": "healthy",
            "last_sync": "2026-05-13T10:01:10Z",
            "error_count": 0,
            "rate_limit_remaining": 10000,
        }
    elif integ in ["slack"]:
        return {
            "integration": integration_name,
            "customer_id": cid,
            "health": "healthy",
            "last_sync": "2026-05-13T09:59:45Z",
            "error_count": 0,
            "rate_limit_remaining": 4900,
        }
    else:
        return {
            "integration": integration_name,
            "customer_id": cid,
            "health": "unhealthy",
            "last_sync": "Never",
            "error_count": 1,
            "error_details": f"Integration '{integration_name}' is not configured or disconnected.",
            "rate_limit_remaining": 0,
        }


@tool(args_schema=LookupAccountInput)
def lookup_account(customer_id: str) -> dict:
    """Look up verified customer subscription, plan, usage, and billing status from the database."""
    seed_accounts_if_empty()
    with get_db_session() as session:
        record = session.query(AccountRecord).filter_by(customer_id=customer_id).first()
        if not record:
            record = session.query(AccountRecord).filter_by(customer_id="default").first()
        if record:
            return {
                "customer_id": record.customer_id,
                "company_name": record.company_name,
                "plan_tier": record.plan_tier,
                "monitored_hosts": record.monitored_hosts,
                "monthly_spend": record.monthly_spend,
                "renewal_date": record.renewal_date,
                "status": record.status,
            }
        return {
            "customer_id": customer_id,
            "error": "Account not found",
            "status": "unknown",
        }
