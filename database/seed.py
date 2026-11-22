"""Seed realistic accounts into the database."""

from database.connection import get_db_session, init_db
from database.models import AccountRecord


INITIAL_ACCOUNTS = [
    {
        "customer_id": "cust_101",
        "company_name": "Acme Corp",
        "plan_tier": "Enterprise",
        "monitored_hosts": 250,
        "monthly_spend": 599.0,
        "renewal_date": "2026-12-31",
        "status": "active",
    },
    {
        "customer_id": "cust_102",
        "company_name": "BetaTech Solutions",
        "plan_tier": "Pro",
        "monitored_hosts": 45,
        "monthly_spend": 199.0,
        "renewal_date": "2026-09-15",
        "status": "active",
    },
    {
        "customer_id": "cust_103",
        "company_name": "CloudScale Inc",
        "plan_tier": "Starter",
        "monitored_hosts": 8,
        "monthly_spend": 49.0,
        "renewal_date": "2026-06-30",
        "status": "active",
    },
    {
        "customer_id": "cust_104",
        "company_name": "DevStudio Global",
        "plan_tier": "Pro",
        "monitored_hosts": 90,
        "monthly_spend": 199.0,
        "renewal_date": "2026-08-01",
        "status": "active",
    },
    {
        "customer_id": "cust_105",
        "company_name": "EnterpriseOne Logistics",
        "plan_tier": "Enterprise",
        "monitored_hosts": 1200,
        "monthly_spend": 1499.0,
        "renewal_date": "2027-01-01",
        "status": "active",
    },
    {
        "customer_id": "default",
        "company_name": "CloudDash Demo Account",
        "plan_tier": "Pro",
        "monitored_hosts": 20,
        "monthly_spend": 199.0,
        "renewal_date": "2026-11-01",
        "status": "active",
    },
]


def seed_accounts_if_empty():
    """Ensure database is initialized and initial accounts exist."""
    init_db()
    with get_db_session() as session:
        count = session.query(AccountRecord).count()
        if count == 0:
            for acc in INITIAL_ACCOUNTS:
                session.add(AccountRecord(**acc))
            session.commit()
