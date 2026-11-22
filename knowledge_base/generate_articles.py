"""Generate 100+ realistic enterprise KB articles across the 5 standard categories.

Categories:
1. troubleshooting (21 articles)
2. billing (21 articles)
3. account (21 articles)
4. faq (21 articles)
5. api_docs (21 articles)

Total generated: 105 articles (KB-021 through KB-125).
Matches the exact KBArticle schema:
- id: KB-xxx
- title: str
- category: str
- tags: list[str]
- content: str (markdown format with sections, code blocks, troubleshooting steps)
- last_updated: str
- applies_to: list[str]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Article blueprints covering enterprise cloud monitoring domain
ARTICLE_SPECS = [
    # --- TROUBLESHOOTING (KB-021 to KB-041) ---
    ("KB-021", "troubleshooting", "Kubernetes DaemonSet Pod CrashLoopBackOff Resolution",
     ["kubernetes", "daemonset", "crashloop", "memory", "troubleshooting"],
     ["Pro", "Enterprise"],
     "Investigating and resolving CrashLoopBackOff errors on CloudDash collector DaemonSets. Common causes include OOMKilled by cgroup memory limits (increase container limits to 512Mi), missing RBAC clusterrolebindings for node metrics, and kernel version incompatibilities with eBPF probes."),

    ("KB-022", "troubleshooting", "Prometheus Scrape Timeout and Target Dropping",
     ["prometheus", "scrape", "timeout", "metrics", "monitoring"],
     ["Starter", "Pro", "Enterprise"],
     "Fixing Prometheus target scrape timeouts. When scrape duration exceeds scrape_interval (default 15s), targets drop. Adjust scrape_timeout, optimize high-cardinality label dimensions, and configure histogram bucket pruning in clouddash-agent.yaml."),

    ("KB-023", "troubleshooting", "GCP Cloud Monitoring Integration Latency",
     ["gcp", "google-cloud", "latency", "stackdriver", "integration"],
     ["Pro", "Enterprise"],
     "Troubleshooting delayed metrics from Google Cloud Monitoring API. GCP metrics exhibit a native 3-5 minute ingestion delay for Compute Engine and Cloud SQL. Configure Stackdriver batching window and verify service account permissions roles/monitoring.viewer."),

    ("KB-024", "troubleshooting", "Azure Monitor Workspace Ingestion Quota Reached",
     ["azure", "quota", "log-analytics", "throttling", "troubleshooting"],
     ["Pro", "Enterprise"],
     "Resolving error 429 Quota Exceeded on Azure Log Analytics workspaces. Check daily cap limits under Cost and Usage, implement table-level data retention filters, and enable CloudDash alert throttling rules to prevent ingestion quota burnout."),

    ("KB-025", "troubleshooting", "Redis Cache Eviction Spikes and Memory Pressure Alerts",
     ["redis", "cache", "eviction", "memory", "alerts"],
     ["Starter", "Pro", "Enterprise"],
     "Diagnosing sudden memory usage spikes and volatile-lru evictions in Redis clusters. Review maxmemory-policy settings, inspect bigkeys with redis-cli --bigkeys, and tune CloudDash alert sensitivity for memory fragmentation ratio > 1.5."),

    ("KB-026", "troubleshooting", "Envoy and Istio Proxy Connection Reset (503 UC)",
     ["envoy", "istio", "mesh", "503", "connection-reset"],
     ["Enterprise"],
     "Troubleshooting 503 Service Unavailable with response flag UC (upstream connection termination) in Istio service mesh. Verify circuit breaker max_connections, idle_timeout in VirtualService destination rules, and keepalive probes."),

    ("KB-027", "troubleshooting", "SSL/TLS Handshake Failures on Agent Webhooks",
     ["ssl", "tls", "certificates", "handshake", "webhooks"],
     ["Starter", "Pro", "Enterprise"],
     "Resolving TLS handshake errors (ERR_SSL_VERSION_OR_CIPHER_MISMATCH) on external webhook endpoints. CloudDash requires TLS 1.2 or TLS 1.3 with valid intermediate certificate chains. Test certificates using openssl s_client -connect."),

    ("KB-028", "troubleshooting", "CoreDNS Resolution Throttling and DNS Query Timeouts",
     ["dns", "coredns", "kubernetes", "networking", "troubleshooting"],
     ["Pro", "Enterprise"],
     "Diagnosing high CoreDNS latency and 5-second DNS timeouts in container workloads. Enable NodeLocal DNSCache, increase CoreDNS replica count based on cluster CPU utilization, and inspect CloudDash DNS latency metrics."),

    ("KB-029", "troubleshooting", "Disk I/O Wait Spikes and High Queue Depth Alerts",
     ["disk", "storage", "iowait", "ebs", "performance"],
     ["Starter", "Pro", "Enterprise"],
     "Analyzing disk I/O bottlenecks and EBS volume bursting exhaustion. Check IOPS burst balance on AWS gp2/gp3 volumes, analyze /proc/diskstats, and migrate high write workloads to provisioned IOPS io2 volumes."),

    ("KB-030", "troubleshooting", "Kafka Consumer Lag Accumulation and Partition Rebalancing",
     ["kafka", "streaming", "consumer-lag", "partitions", "troubleshooting"],
     ["Pro", "Enterprise"],
     "Mitigating severe Kafka consumer lag during partition rebalance storms. Tune max.poll.interval.ms, increase consumer group workers, and set CloudDash consumer lag threshold rules with 3-minute sustained breach windows."),

    ("KB-031", "troubleshooting", "Elasticsearch Heap Memory Exhaustion and Circuit Breakers",
     ["elasticsearch", "heap", "jvm", "memory", "troubleshooting"],
     ["Pro", "Enterprise"],
     "Resolving Elasticsearch parent circuit breaker exceptions (Data too large). Verify that ES_JAVA_OPTS heap size does not exceed 50% of system RAM or 31GB, force merge historical read-only indices, and close unused indices."),

    ("KB-032", "troubleshooting", "gRPC Status Code 14 UNAVAILABLE in Microservices",
     ["grpc", "status-14", "microservices", "networking", "troubleshooting"],
     ["Pro", "Enterprise"],
     "Investigating intermittent gRPC 14 UNAVAILABLE errors. Causes include upstream load balancer keepalive mismatches, connection draining on Kubernetes pod termination, and missing grpc_retry interceptors in client code."),

    ("KB-033", "troubleshooting", "PostgreSQL Connection Pool Exhaustion and 53300 Errors",
     ["postgresql", "database", "connection-pool", "pgbouncer", "troubleshooting"],
     ["Starter", "Pro", "Enterprise"],
     "Resolving PostgreSQL error 'FATAL: remaining connection slots are reserved for non-replication superuser connections'. Deploy PgBouncer connection pooler in transaction mode and configure CloudDash active connection count alerts at 80% capacity."),

    ("KB-034", "troubleshooting", "Docker Container OOMKilled Error Code 137",
     ["docker", "oomkilled", "exit-137", "containers", "memory"],
     ["Starter", "Pro", "Enterprise"],
     "Diagnosing container exits with code 137 (OOMKilled). Inspect docker inspect or dmesg for kernel out-of-memory killer logs, review JVM heap versus container limits, and configure proactive memory trajectory alerts."),

    ("KB-035", "troubleshooting", "Nginx 502 Bad Gateway with Upstream Timeout",
     ["nginx", "502", "bad-gateway", "upstream", "proxy"],
     ["Starter", "Pro", "Enterprise"],
     "Resolving 502 Bad Gateway responses from reverse proxy layers. Check upstream backend health checks, increase proxy_connect_timeout and proxy_read_timeout in nginx.conf, and monitor upstream response time percentiles (p95/p99)."),

    ("KB-036", "troubleshooting", "OpenTelemetry Collector Export Failures and Dropped Spans",
     ["opentelemetry", "otel", "spans", "export", "troubleshooting"],
     ["Pro", "Enterprise"],
     "Fixing dropped traces and spans in OpenTelemetry collector pipelines. Check queue_size in batch processor, adjust max_elapsed_time in retry_on_failure, and verify CloudDash OTLP endpoint authorization tokens."),

    ("KB-037", "troubleshooting", "AWS IAM STS AssumeRole AccessDenied in Cross-Account Setup",
     ["aws", "iam", "sts", "cross-account", "security"],
     ["Enterprise"],
     "Resolving sts:AssumeRole AccessDenied errors when CloudDash monitors multiple AWS accounts. Ensure the trust policy in the target account specifies the correct ExternalId and allows sts:AssumeRole from CloudDash principal ARN."),

    ("KB-038", "troubleshooting", "RabbitMQ Unacknowledged Message Spikes and Channel Freezing",
     ["rabbitmq", "queues", "unack", "memory-alarm", "troubleshooting"],
     ["Pro", "Enterprise"],
     "Diagnosing blocked RabbitMQ channels due to high memory watermarks or missing consumer basic.ack acknowledgments. Set consumer prefetch count (basic.qos) to 50-100 and configure queue depth escalation thresholds."),

    ("KB-039", "troubleshooting", "Network Latency and Packet Loss in Multi-Region VPC Peering",
     ["networking", "vpc", "latency", "packet-loss", "aws"],
     ["Enterprise"],
     "Diagnosing inter-region packet loss across AWS VPC peering connections. Use CloudDash synthetic ping probes, verify MTU path discovery (jumbo frames vs 1500 standard MTU), and inspect Transit Gateway attachment metrics."),

    ("KB-040", "troubleshooting", "Logstash Pipeline Stalls and Backpressure on Beats Agents",
     ["logstash", "beats", "backpressure", "pipelines", "troubleshooting"],
     ["Pro", "Enterprise"],
     "Troubleshooting pipeline stalls in Logstash workers. Check persistent queue disk space, inspect slow grok filter regular expressions, and adjust pipeline.workers and pipeline.batch.size to alleviate worker starvation."),

    ("KB-041", "troubleshooting", "High CPU Throttling on Burstable EC2 T3/T4g Instances",
     ["ec2", "cpu-credits", "throttling", "performance", "aws"],
     ["Starter", "Pro"],
     "Diagnosing severe performance degradation on AWS T3/T4g instances when CPU credit balance reaches zero. Enable T-instance unlimited mode or upgrade to non-burstable compute instances (m6i/c6i) for sustained production workloads."),

    # --- BILLING (KB-042 to KB-062) ---
    ("KB-042", "billing", "Annual Enterprise Contract Commitments and True-Up Terms",
     ["billing", "enterprise", "annual-contract", "true-up", "commitments"],
     ["Enterprise"],
     "Overview of annual enterprise subscription contracts, volume discounts, and quarterly true-up calculations for hosts and custom metrics exceeding contracted thresholds."),

    ("KB-043", "billing", "Invoice Dispute and Credit Adjustment Procedure",
     ["billing", "invoice", "dispute", "credit", "refund"],
     ["Starter", "Pro", "Enterprise"],
     "How to initiate an invoice dispute or request credit adjustments for service downtime or billing errors. Disputes must be submitted within 30 days of the invoice date."),

    ("KB-044", "billing", "Credit Card Expiration and Automated Payment Retries",
     ["billing", "credit-card", "payment-retry", "dunning", "failure"],
     ["Starter", "Pro"],
     "Automated payment retry schedule (smart dunning): retries occur on day 1, 3, 7, and 14 following a failed card charge before account suspension enters grace period."),

    ("KB-045", "billing", "Tax Exemption Certificate Upload and VAT Invalidation",
     ["billing", "tax", "vat", "gst", "exemption"],
     ["Starter", "Pro", "Enterprise"],
     "Uploading state tax exemption certificates (US) and validating EU VAT / UK VAT / Australian GST identification numbers for tax-exempt invoicing."),

    ("KB-046", "billing", "Multi-Currency Billing Options and Exchange Rate Guarantees",
     ["billing", "currency", "usd", "eur", "gbp"],
     ["Pro", "Enterprise"],
     "Supported billing currencies (USD, EUR, GBP, CAD, AUD) and exchange rate handling for annual contracts and monthly card billing."),

    ("KB-047", "billing", "Custom Metric Pricing Overages and Tiered Rate Cards",
     ["billing", "custom-metrics", "pricing", "overages", "rate-card"],
     ["Pro", "Enterprise"],
     "Detailed breakdown of custom metric pricing: first 10,000 metrics included on Pro; additional metrics billed at $0.05 per metric/month with tiered discounts above 100k metrics."),

    ("KB-048", "billing", "Reserved Seat Licensing and Add-On User Pricing",
     ["billing", "seats", "users", "licensing", "pricing"],
     ["Starter", "Pro", "Enterprise"],
     "Purchasing additional user seats beyond plan allowances: Starter includes 3 seats ($10/seat/mo add-on), Pro includes 10 seats ($15/seat/mo add-on), Enterprise includes unlimited seats."),

    ("KB-049", "billing", "Purchase Order (PO) Invoicing and Net-30 Payment Terms",
     ["billing", "purchase-order", "po", "net-30", "wire-transfer"],
     ["Enterprise"],
     "Requirements for Net-30 purchase order (PO) invoicing on Enterprise plans. Minimum annual contract value of $5,000 required for manual vendor setup."),

    ("KB-050", "billing", "Billing Admin Role Delegation and Notification Routing",
     ["billing", "rbac", "billing-admin", "roles", "invoices"],
     ["Pro", "Enterprise"],
     "Assigning the Billing Admin role to finance team members without granting access to production cloud infrastructure or monitoring dashboards."),

    ("KB-051", "billing", "Payment Failure Grace Period and Service Suspension Schedule",
     ["billing", "suspension", "grace-period", "dunning", "policy"],
     ["Starter", "Pro", "Enterprise"],
     "Service suspension timelines following uncollected payments: 14 days full access, 7 days read-only dashboard access, followed by data ingestion pause."),

    ("KB-052", "billing", "Nonprofit and Open-Source Project Discount Programs",
     ["billing", "discounts", "nonprofit", "open-source", "education"],
     ["Starter", "Pro"],
     "Eligibility guidelines and application process for verified 501(c)(3) nonprofits and open-source public repositories seeking 50% subscription discounts."),

    ("KB-053", "billing", "Early Termination Policy for Multi-Year Enterprise Contracts",
     ["billing", "enterprise", "termination", "contracts", "cancellation"],
     ["Enterprise"],
     "Terms and conditions for early termination of multi-year enterprise agreements, buyout fees, and transition service periods."),

    ("KB-054", "billing", "Understanding Prorated Charges on Mid-Month Seat Additions",
     ["billing", "proration", "seats", "invoices", "calculation"],
     ["Starter", "Pro"],
     "How prorated charges are computed down to the minute when adding user seats or add-on modules during an active billing cycle."),

    ("KB-055", "billing", "Self-Service Invoice Downloading and Receipt History",
     ["billing", "invoices", "pdf", "receipts", "dashboard"],
     ["Starter", "Pro", "Enterprise"],
     "Step-by-step instructions to view, download, and email PDF tax invoices and receipts from the Settings > Billing portal."),

    ("KB-056", "billing", "Cloud Marketplace Billing (AWS Marketplace & GCP Marketplace)",
     ["billing", "aws-marketplace", "gcp-marketplace", "procurement", "consolidated-billing"],
     ["Enterprise"],
     "Subscribing to CloudDash through AWS Marketplace or Google Cloud Marketplace to draw down against existing committed cloud spend (EDP / Committed Use)."),

    ("KB-057", "billing", "Synthetic Monitoring Check Pricing and Run Frequency Costs",
     ["billing", "synthetic-checks", "monitoring", "pricing", "overages"],
     ["Pro", "Enterprise"],
     "Cost structure for synthetic HTTP and browser-based uptime checks: 50 checks included on Pro; additional check runs billed at $0.001 per run."),

    ("KB-058", "billing", "Log Ingestion Volume Billing and GB/Day Overage Calculator",
     ["billing", "logs", "ingestion", "storage", "pricing"],
     ["Pro", "Enterprise"],
     "Calculating log ingestion fees beyond plan quotas: Pro includes 50GB/month; excess log volume billed at $0.25/GB ingested with 30-day retention."),

    ("KB-059", "billing", "Updating Billing Address, Company Legal Name, and Tax ID",
     ["billing", "address", "company-info", "tax-id", "settings"],
     ["Starter", "Pro", "Enterprise"],
     "How to update your company's registered legal name, billing address, and corporate tax identifier on subsequent PDF invoices."),

    ("KB-060", "billing", "Prepaid Account Credits and Depletion Tracking",
     ["billing", "credits", "prepaid", "balance", "drawdown"],
     ["Enterprise"],
     "Managing prepaid account credits, viewing draw-down rates across compute hosts, and setting low-credit warning notifications at 20% remaining."),

    ("KB-061", "billing", "Downgrading to Free Developer Tier Policy and Data Pruning",
     ["billing", "downgrade", "free-tier", "data-retention", "policy"],
     ["Starter"],
     "What happens when downgrading from a paid Starter plan to the Free Developer tier: retention drops to 24 hours and host count reduces to 1."),

    ("KB-062", "billing", "ACH Direct Debit Failure Codes and Bank Authorization",
     ["billing", "ach", "direct-debit", "bank-transfer", "failure"],
     ["Enterprise"],
     "Resolving ACH direct debit return codes (R01 insufficient funds, R02 account closed, R08 payment stopped) and submitting bank debit mandates."),

    # --- ACCOUNT & SECURITY (KB-063 to KB-083) ---
    ("KB-063", "account", "Okta SCIM Automated User Provisioning and Deprovisioning",
     ["account", "okta", "scim", "provisioning", "security"],
     ["Enterprise"],
     "Configuring Okta SCIM 2.0 integration for automated team member onboarding, role assignment, and immediate deprovisioning on employee offboarding."),

    ("KB-064", "account", "Azure Active Directory Group Mapping to CloudDash Roles",
     ["account", "azure-ad", "entra-id", "rbac", "groups"],
     ["Enterprise"],
     "Mapping Azure AD / Microsoft Entra security groups directly to CloudDash Owner, Admin, Member, and Read-Only roles using SAML attribute assertions."),

    ("KB-065", "account", "Multi-Factor Authentication (MFA/2FA) Enforcement Policy",
     ["account", "mfa", "2fa", "security", "authentication"],
     ["Starter", "Pro", "Enterprise"],
     "Enforcing mandatory Time-based One-Time Password (TOTP) multi-factor authentication across all team members in the organization."),

    ("KB-066", "account", "Dashboard Session Timeout and Inactivity Invalidation",
     ["account", "session-timeout", "security", "compliance", "settings"],
     ["Pro", "Enterprise"],
     "Configuring maximum session duration and idle timeout limits (15 minutes to 24 hours) for SOC2 compliance across web dashboard sessions."),

    ("KB-067", "account", "IP Allowlisting for Web Dashboard and API Access",
     ["account", "ip-allowlist", "security", "firewall", "enterprise"],
     ["Enterprise"],
     "Restricting dashboard logins and API token requests to designated corporate CIDR IP blocks and VPN gateways."),

    ("KB-068", "account", "Enterprise SSO Domain Verification via DNS TXT Record",
     ["account", "domain-verification", "dns", "sso", "security"],
     ["Enterprise"],
     "Validating corporate domain ownership by publishing a unique CloudDash verification token to your DNS TXT records before enforcing SAML SSO."),

    ("KB-069", "account", "Service Account Creation and Non-Expiring Bot Tokens",
     ["account", "service-accounts", "api-tokens", "automation", "security"],
     ["Pro", "Enterprise"],
     "Creating headless service accounts for CI/CD pipelines, Terraform deployments, and automated alerting integrations with scoped API permissions."),

    ("KB-070", "account", "Audit Log Streaming to Amazon S3 and Google Cloud Storage",
     ["account", "audit-logs", "s3", "gcs", "compliance"],
     ["Enterprise"],
     "Configuring automated real-time export of organization audit logs to an Amazon S3 bucket or GCP Cloud Storage bucket in NDJSON format."),

    ("KB-071", "account", "Team Workspace Isolation and Multi-Tenant Project Segregation",
     ["account", "workspaces", "teams", "isolation", "projects"],
     ["Enterprise"],
     "Setting up isolated team workspaces under a single parent enterprise billing account to partition metrics, dashboards, and alerts between departments."),

    ("KB-072", "account", "Configuring Granular Dashboard Read/Write ACLs",
     ["account", "dashboard-acls", "permissions", "rbac", "security"],
     ["Pro", "Enterprise"],
     "Restricting dashboard editing and deletion rights to specific teams or roles while allowing broad read-only viewer access across the organization."),

    ("KB-073", "account", "Emergency Account Recovery and Break-Glass Administrator",
     ["account", "emergency-recovery", "break-glass", "security", "admin"],
     ["Enterprise"],
     "Designating and configuring emergency break-glass administrator credentials when SSO Identity Providers (Okta/Azure AD) suffer third-party outages."),

    ("KB-074", "account", "Password Policy Complexity and Expiration Configuration",
     ["account", "password-policy", "security", "compliance", "authentication"],
     ["Pro", "Enterprise"],
     "Setting password length, special character complexity, 90-day password rotation requirements, and disallowing reused historical passwords."),

    ("KB-075", "account", "Account Deletion, Data Purging, and GDPR Right to Erasure",
     ["account", "gdpr", "account-deletion", "privacy", "compliance"],
     ["Starter", "Pro", "Enterprise"],
     "Submitting a formal account termination and complete data erasure request in compliance with GDPR Article 17 and CCPA requirements."),

    ("KB-076", "account", "SAML Single Logout (SLO) Configuration Guide",
     ["account", "saml", "slo", "single-logout", "security"],
     ["Enterprise"],
     "Configuring SAML 2.0 Single Logout (SLO) to ensure terminating a CloudDash session terminates the corresponding IdP session and vice versa."),

    ("KB-077", "account", "Custom RBAC Role Definition and Scoped Permission Matrix",
     ["account", "rbac", "custom-roles", "permissions", "security"],
     ["Enterprise"],
     "Creating bespoke roles (e.g. SRE Responder, Compliance Auditor) by selecting fine-grained permissions from CloudDash's 40+ permission verbs."),

    ("KB-078", "account", "Managing Multiple CloudDash Organizations with One Email",
     ["account", "multi-org", "organization-switch", "accounts", "settings"],
     ["Starter", "Pro", "Enterprise"],
     "Switching between different customer organizations and development environments seamlessly using a single unified email address."),

    ("KB-079", "account", "Suspicious Login Alerts and Geolocation Anomaly Detection",
     ["account", "security-alerts", "anomaly-detection", "logins", "security"],
     ["Pro", "Enterprise"],
     "How CloudDash detects and notifies administrators of impossible-travel logins and unrecognized device fingerprints."),

    ("KB-080", "account", "API Key Expiration Policy and Automated Rotation Alerts",
     ["account", "api-keys", "key-rotation", "security", "compliance"],
     ["Pro", "Enterprise"],
     "Enforcing mandatory 90-day or 180-day expiration windows on all issued API tokens with automated Slack and email warning notices."),

    ("KB-081", "account", "Configuring Webhook Notification Endpoints for Audit Events",
     ["account", "audit-webhooks", "siem", "notifications", "security"],
     ["Enterprise"],
     "Streaming security audit events (user created, role modified, API key reset) to external incident response webhooks and SOAR platforms."),

    ("KB-082", "account", "Data Residency Region Selection (US, EU, APAC)",
     ["account", "data-residency", "eu-gdpr", "apac", "compliance"],
     ["Enterprise"],
     "Selecting storage regions (US-East, EU-Frankfurt, APAC-Tokyo) for customer metric data and log storage to satisfy sovereign data compliance."),

    ("KB-083", "account", "Transferring Organization Ownership to Another Administrator",
     ["account", "ownership-transfer", "admin", "roles", "organization"],
     ["Starter", "Pro", "Enterprise"],
     "How an existing account Owner can safely transfer the primary Owner designation to another designated Admin user."),

    # --- FAQ & GENERAL (KB-084 to KB-104) ---
    ("KB-084", "faq", "SOC2 Type II Compliance Reports and Security Documentation",
     ["faq", "soc2", "compliance", "security", "audits"],
     ["Starter", "Pro", "Enterprise"],
     "Requesting CloudDash's latest annual SOC 2 Type II audit report, penetration testing executive summary, and security whitepaper under NDA."),

    ("KB-085", "faq", "ISO 27001 Certification and Information Security Policies",
     ["faq", "iso27001", "certification", "compliance", "security"],
     ["Enterprise"],
     "Details on CloudDash's ISO/IEC 27001:2022 certification scope, risk management controls, and third-party accreditation bodies."),

    ("KB-086", "faq", "Standard Service Level Agreement (SLA) and Uptime Guarantees",
     ["faq", "sla", "uptime", "availability", "guarantee"],
     ["Pro", "Enterprise"],
     "CloudDash SLA guarantees: 99.9% uptime on Pro, 99.95% on Enterprise, service credit penalty schedules, and excluded maintenance exclusions."),

    ("KB-087", "faq", "Subscribing to Real-Time CloudDash Platform Status Notifications",
     ["faq", "status-page", "notifications", "outages", "incident"],
     ["Starter", "Pro", "Enterprise"],
     "How to subscribe to email, SMS, and webhook alerts from status.clouddash.io for planned maintenance windows and service disruptions."),

    ("KB-088", "faq", "Agentless vs Agent-Based Cloud Monitoring Comparison",
     ["faq", "architecture", "agentless", "agent", "monitoring"],
     ["Starter", "Pro", "Enterprise"],
     "Comparing agentless monitoring (via AWS CloudWatch / Azure Monitor APIs) with host-based clouddash-agent (eBPF, process-level telemetry, kernel stats)."),

    ("KB-089", "faq", "Cloud Provider Network Egress Costs Incurred by CloudDash",
     ["faq", "network-egress", "costs", "bandwidth", "aws"],
     ["Starter", "Pro", "Enterprise"],
     "Estimating and minimizing cloud provider network egress bandwidth fees when transmitting metrics to CloudDash ingestion endpoints."),

    ("KB-090", "faq", "Maintenance Window Schedules and Alert Suppression Rules",
     ["faq", "maintenance", "alert-suppression", "scheduled", "downtime"],
     ["Starter", "Pro", "Enterprise"],
     "Scheduled maintenance policy: CloudDash deploys zero-downtime rolling updates; scheduled infrastructure work occurs Sundays 02:00-04:00 UTC."),

    ("KB-091", "faq", "Telemetry Ingestion Endpoints and IP Whitelisting for Outbound Firewalls",
     ["faq", "firewall", "ip-addresses", "endpoints", "networking"],
     ["Starter", "Pro", "Enterprise"],
     "Official list of static IPv4 ranges and domain names (ingest.clouddash.io:443) required for outbound egress firewall rules."),

    ("KB-092", "faq", "Supported Operating Systems and Linux Kernel Compatibility",
     ["faq", "linux", "kernel", "os-support", "compatibility"],
     ["Starter", "Pro", "Enterprise"],
     "Supported host OS platforms: Ubuntu 20.04+, Debian 11+, RHEL/Rocky 8+, Amazon Linux 2023, and Windows Server 2019+; minimum kernel 5.4 for eBPF."),

    ("KB-093", "faq", "Data Encryption at Rest and in Transit Standards",
     ["faq", "encryption", "kms", "tls", "security"],
     ["Starter", "Pro", "Enterprise"],
     "Details on AES-256-GCM encryption at rest, TLS 1.3 encryption in transit, and optional customer-managed AWS KMS keys (CMEK) on Enterprise."),

    ("KB-094", "faq", "Submitting Feature Requests and Roadmap Visibility",
     ["faq", "feature-requests", "roadmap", "feedback", "product"],
     ["Starter", "Pro", "Enterprise"],
     "How customers can submit, vote on, and track upcoming product features through our public portal and customer advisory board."),

    ("KB-095", "faq", "Migrating Dashboards and Alerts from Datadog to CloudDash",
     ["faq", "migration", "datadog", "import", "dashboards"],
     ["Pro", "Enterprise"],
     "Using our automated migration utility (clouddash-migrate) to convert Datadog dashboard JSON and alert monitor rules into CloudDash definitions."),

    ("KB-096", "faq", "Migrating from Prometheus/Grafana to CloudDash",
     ["faq", "grafana", "prometheus", "migration", "dashboards"],
     ["Starter", "Pro", "Enterprise"],
     "Importing Grafana JSON dashboard definitions and PromQL alerting queries directly into CloudDash native monitoring boards."),

    ("KB-097", "faq", "Data Processing Agreement (DPA) and Standard Contractual Clauses",
     ["faq", "dpa", "gdpr", "compliance", "privacy"],
     ["Starter", "Pro", "Enterprise"],
     "Accessing, reviewing, and electronically executing the CloudDash GDPR Data Processing Agreement (DPA) and EU Standard Contractual Clauses (SCCs)."),

    ("KB-098", "faq", "Disaster Recovery (DR) and Multi-Region Redundancy Architecture",
     ["faq", "disaster-recovery", "rpo", "rto", "architecture"],
     ["Enterprise"],
     "CloudDash business continuity specifications: RPO (Recovery Point Objective) < 15 minutes, RTO (Recovery Time Objective) < 1 hour with active-passive region failover."),

    ("KB-099", "faq", "HIPAA Compliance and Business Associate Agreements (BAA)",
     ["faq", "hipaa", "baa", "healthcare", "compliance"],
     ["Enterprise"],
     "Executing a Business Associate Agreement (BAA) for HIPAA compliance, PHI log scrubbing rules, and dedicated infrastructure options."),

    ("KB-100", "faq", "Custom Webhook Payload Templating and Variable Substitution",
     ["faq", "webhooks", "templates", "jinja", "formatting"],
     ["Starter", "Pro", "Enterprise"],
     "Using Jinja2 variable substitution in webhook payload bodies ({{ alert.id }}, {{ metric.value }}, {{ timestamp }}, {{ severity }})."),

    ("KB-101", "faq", "Alert Fatigue Reduction and Flapping Detection Algorithms",
     ["faq", "alert-fatigue", "flapping", "noise-reduction", "thresholds"],
     ["Pro", "Enterprise"],
     "How CloudDash detects oscillating alert states (flapping) and automatically mutes repetitive notifications until metrics stabilize."),

    ("KB-102", "faq", "Differences Between Metrics, Logs, and Traces in CloudDash",
     ["faq", "telemetry", "metrics", "logs", "traces"],
     ["Starter", "Pro", "Enterprise"],
     "Educational primer distinguishing numerical metrics, event logs, and distributed request traces across modern observability stacks."),

    ("KB-103", "faq", "Browser Support and Recommended Display Resolutions",
     ["faq", "browsers", "chrome", "firefox", "display"],
     ["Starter", "Pro", "Enterprise"],
     "Supported desktop browsers (Chrome 110+, Firefox 115+, Safari 16+, Edge 110+) and optimized resolutions for NOC monitor wallboard displays."),

    ("KB-104", "faq", "Dedicated Support Channels and Technical Account Managers (TAM)",
     ["faq", "support", "tam", "enterprise-support", "sla"],
     ["Enterprise"],
     "Enterprise tier benefits: dedicated Slack/Teams connect channels, 15-minute critical response SLA, and assigned Technical Account Managers."),

    # --- API DOCS & INTEGRATION (KB-105 to KB-125) ---
    ("KB-105", "api_docs", "OpenTelemetry (OTLP) gRPC and HTTP Ingestion API",
     ["api", "opentelemetry", "otlp", "grpc", "endpoints"],
     ["Pro", "Enterprise"],
     "Sending traces and metrics directly to CloudDash using standard OpenTelemetry OTLP/gRPC (port 4317) and OTLP/HTTP (port 4318) endpoints with authorization headers."),

    ("KB-106", "api_docs", "Batch Custom Metric Ingestion POST /api/v1/metrics/batch",
     ["api", "metrics", "batch", "ingestion", "rest"],
     ["Starter", "Pro", "Enterprise"],
     "Pushing high-volume custom metric samples via POST /api/v1/metrics/batch. Supports up to 1,000 data points per request with timestamp, name, tags, and float value."),

    ("KB-107", "api_docs", "Querying Historical Metric Aggregations GET /api/v1/metrics/query",
     ["api", "query", "aggregation", "time-series", "metrics"],
     ["Starter", "Pro", "Enterprise"],
     "Querying time-series metric data with downsampling functions (avg, max, min, sum, p95, p99) over custom time windows via GET /api/v1/metrics/query."),

    ("KB-108", "api_docs", "Programmatic Alert Rule Management API CRUD Endpoints",
     ["api", "alerts", "rules", "crud", "automation"],
     ["Starter", "Pro", "Enterprise"],
     "Creating, reading, updating, and deleting alert rules programmatically using GET, POST, PUT, DELETE on /api/v1/alerts/rules."),

    ("KB-109", "api_docs", "Alert Notification Channels API /api/v1/notifications/channels",
     ["api", "notifications", "slack", "pagerduty", "channels"],
     ["Starter", "Pro", "Enterprise"],
     "Managing notification channel configurations for Slack webhooks, PagerDuty integration keys, Opsgenie, and email distribution lists via API."),

    ("KB-110", "api_docs", "Incident and Alert History Extraction GET /api/v1/alerts/events",
     ["api", "incidents", "events", "history", "export"],
     ["Starter", "Pro", "Enterprise"],
     "Exporting historical alert trigger events, duration, severity, and resolution timestamps for SLA reporting and post-mortem analysis."),

    ("KB-111", "api_docs", "Dashboard Export and Import as Code POST /api/v1/dashboards/import",
     ["api", "dashboards", "dashboard-as-code", "json", "export"],
     ["Pro", "Enterprise"],
     "Exporting dashboards as portable JSON definitions and importing them programmatically into staging and production accounts."),

    ("KB-112", "api_docs", "Terraform Provider for CloudDash Resource Provisioning",
     ["api", "terraform", "iac", "provider", "automation"],
     ["Pro", "Enterprise"],
     "Using the official terraform-provider-clouddash to manage alert rules, notification channels, synthetic checks, and team RBAC as code."),

    ("KB-113", "api_docs", "Synthetic Monitoring Check Creation API POST /api/v1/synthetics",
     ["api", "synthetics", "uptime", "http-checks", "monitoring"],
     ["Pro", "Enterprise"],
     "Configuring global HTTP, TLS certificate, and DNS synthetic checks with geographic probe location selection via POST /api/v1/synthetics/checks."),

    ("KB-114", "api_docs", "Log Ingestion and Bulk Search API POST /api/v1/logs",
     ["api", "logs", "ingestion", "search", "endpoints"],
     ["Pro", "Enterprise"],
     "Ingesting structured JSON log entries and performing Lucene-style full-text queries with timestamp filtering via POST /api/v1/logs/search."),

    ("KB-115", "api_docs", "Tagging and Host Metadata Enrichment API PUT /api/v1/hosts/tags",
     ["api", "tags", "hosts", "metadata", "enrichment"],
     ["Starter", "Pro", "Enterprise"],
     "Attaching custom metadata key-value tags (env:prod, service:checkout, region:us-east-1) to registered infrastructure hosts via API."),

    ("KB-116", "api_docs", "Maintenance Window Scheduling API /api/v1/maintenance",
     ["api", "maintenance", "schedules", "alert-suppression", "automation"],
     ["Starter", "Pro", "Enterprise"],
     "Programmatically scheduling recurring or one-off alert suppression windows during deployment pipelines using POST /api/v1/maintenance."),

    ("KB-117", "api_docs", "Audit Log Retrieval and SIEM Integration API GET /api/v1/audit",
     ["api", "audit", "siem", "splunk", "security"],
     ["Enterprise"],
     "Querying administrative security audit trails by date range, actor email, and action verb for automated Splunk/Datadog SIEM ingestion."),

    ("KB-118", "api_docs", "User Management and Team Membership API /api/v1/teams/users",
     ["api", "users", "invitations", "team-management", "rbac"],
     ["Pro", "Enterprise"],
     "Inviting users, modifying assigned RBAC roles, and removing departing team members via REST endpoints under /api/v1/teams/members."),

    ("KB-119", "api_docs", "Webhook Secret and HMAC-SHA256 Signature Rotation API",
     ["api", "webhooks", "hmac", "signature-rotation", "security"],
     ["Pro", "Enterprise"],
     "Rotating webhook shared secrets and testing payload signatures without interrupting live alert deliveries via POST /api/v1/webhooks/rotate."),

    ("KB-120", "api_docs", "Kubernetes Operator for CloudDash Custom Resource Definitions (CRDs)",
     ["api", "kubernetes", "operator", "crd", "gitops"],
     ["Pro", "Enterprise"],
     "Managing CloudDash alert monitors as native Kubernetes Custom Resources (CloudDashAlertRule CRD) in GitOps ArgoCD/Flux pipelines."),

    ("KB-121", "api_docs", "Anomalous Metric Detection API GET /api/v1/analytics/anomalies",
     ["api", "anomaly-detection", "ml", "metrics", "analytics"],
     ["Pro", "Enterprise"],
     "Querying machine-learning detected metric anomalies and baseline forecasting bands via GET /api/v1/analytics/anomalies."),

    ("KB-122", "api_docs", "GraphQL API Reference and Schema Introspection /api/graphql",
     ["api", "graphql", "schema", "queries", "mutations"],
     ["Enterprise"],
     "Querying CloudDash unified metrics, incidents, and infrastructure topology through the GraphQL endpoint at /api/graphql."),

    ("KB-123", "api_docs", "Python SDK Async Client and Streaming Telemetry Guide",
     ["api", "sdk", "python", "asyncio", "streaming"],
     ["Starter", "Pro", "Enterprise"],
     "Using AsyncCloudDashClient in Python 3.10+ for asynchronous metric queuing and background flush workers in FastAPI and asyncio applications."),

    ("KB-124", "api_docs", "Go SDK (clouddash-go) Installation and High-Performance Examples",
     ["api", "sdk", "go", "golang", "high-performance"],
     ["Starter", "Pro", "Enterprise"],
     "Integrating the official Go client library (github.com/clouddash/clouddash-go) with non-blocking channel buffers and zero memory allocations."),

    ("KB-125", "api_docs", "Rate Limit Status and Quota Headers Reference",
     ["api", "rate-limits", "headers", "http-429", "reference"],
     ["Starter", "Pro", "Enterprise"],
     "Comprehensive reference of CloudDash rate-limiting response headers: X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset-After."),
]


def generate_article_content(spec: tuple) -> dict:
    """Generate a rich, multi-section markdown KB article matching the spec."""
    kb_id, category, title, tags, applies_to, summary = spec

    content = f"""# {title}

{summary}

## Overview & Architecture

CloudDash provides enterprise-grade infrastructure monitoring for distributed cloud workloads. This article provides technical guidance on configuring, verifying, and troubleshooting **{title}**.

### Key Technical Specifications
- **Category**: `{category}`
- **Supported Plans**: {', '.join(applies_to)}
- **Component ID**: `{kb_id}`
- **Keywords**: {', '.join(tags)}

## Step-by-Step Implementation Guide

Follow these instructions to configure and operationalize this capability within your environment:

1. **Verify Prerequisites**:
   Ensure you have appropriate administrative permissions and your CloudDash account is on an eligible plan ({', '.join(applies_to)}).

2. **Access the Configuration Interface**:
   Navigate to the relevant settings section in the CloudDash dashboard or initialize your API client with your active Bearer token.

3. **Configure Parameter Settings**:
   Apply the appropriate configuration values tailored to your infrastructure requirements:
   ```yaml
   # Configuration snippet for {kb_id}
   service:
     enabled: true
     category: "{category}"
     target: "{title}"
     evaluation_window: "5m"
     alert_threshold: 85
     retry_attempts: 3
     backoff_multiplier: 1.5
   ```

4. **Validate Live Telemetry**:
   Check integration logs and status badges to verify data stream continuity and absence of connectivity errors.

## Common Error Codes and Troubleshooting

| Error Code | Root Cause | Recommended Remediation |
| :--- | :--- | :--- |
| `ERR_AUTH_EXPIRED` | Expired or revoked API key | Re-generate key under Settings > API Keys [KB-010] |
| `ERR_CONFIG_INVALID` | Malformed parameter payload | Verify YAML/JSON syntax against schema documentation |
| `ERR_RATE_LIMIT_EXCEEDED` | Request frequency exceeds tier limit | Implement exponential backoff or upgrade plan tier [KB-014] |
| `ERR_UPSTREAM_TIMEOUT` | Downstream service latency | Increase timeout thresholds or inspect network connectivity |

## Policy Details & Best Practices

- **Retention & Compliance**: All associated metrics adhere to CloudDash's standard data retention policies ({'365 days on Enterprise' if 'Enterprise' in applies_to else '30 days on Pro'}).
- **Audit Logging**: All configuration changes trigger immutable audit log entries visible under Organization Settings.
- **Support Escalation**: If issues persist beyond these standard steps, contact CloudDash technical engineering via the escalation protocol.
"""

    return {
        "id": kb_id,
        "title": title,
        "category": category,
        "tags": tags,
        "content": content,
        "last_updated": "2026-05-01",
        "applies_to": applies_to,
    }


def generate_all_articles(output_dir: str | None = None) -> list[str]:
    """Generate and write all 105 articles to disk."""
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "articles")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    generated_files = []
    print(f"Generating {len(ARTICLE_SPECS)} new KB articles in {out_path}...")

    for spec in ARTICLE_SPECS:
        article_data = generate_article_content(spec)
        file_path = out_path / f"{article_data['id']}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(article_data, f, indent=2)
        generated_files.append(str(file_path))

    print(f"Successfully generated {len(generated_files)} articles (KB-021 through KB-125).")
    return generated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 100+ Enterprise KB articles")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save generated articles")
    args = parser.parse_args()
    generate_all_articles(args.output_dir)
