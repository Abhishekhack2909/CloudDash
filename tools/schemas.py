"""Pydantic schemas for agent tool calling with strict validation."""

from typing import Optional
from pydantic import BaseModel, Field


class CreateSupportTicketInput(BaseModel):
    """Input schema for creating a support ticket."""

    title: str = Field(..., min_length=3, max_length=256, description="Short summary of the customer's issue")
    description: str = Field(..., min_length=5, description="Detailed description of the issue or error encountered")
    priority: str = Field(default="medium", description="Ticket priority: low, medium, high, or critical")
    category: str = Field(default="technical", description="Ticket category: technical, billing, account, or general")


class CheckIntegrationStatusInput(BaseModel):
    """Input schema for checking integration health."""

    integration_name: str = Field(..., min_length=2, max_length=64, description="Integration name (e.g. AWS, Slack, Kubernetes, Datadog, CloudWatch)")
    customer_id: Optional[str] = Field(default=None, description="Optional customer account ID")


class LookupAccountInput(BaseModel):
    """Input schema for looking up customer account and subscription details."""

    customer_id: str = Field(..., min_length=1, max_length=64, description="Unique customer account identifier (e.g. cust_101, cust_102)")
