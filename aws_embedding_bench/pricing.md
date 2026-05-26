# AWS EC2 pricing — TEI embedding bench instances (eu-central-1, Linux)

_Looked up: 2026-05-21. On-demand prices change rarely; RIs change with AWS pricing-page refreshes. All values are for Linux/UNIX, shared tenancy, no preinstalled software, in `eu-central-1` (Frankfurt). RI columns are **Standard, No-Upfront** (`OfferingClass=standard`, `PurchaseOption=No Upfront`). Monthly = hourly × 730 h._

| Instance      | GPU / accel.       | On-demand $/hr | On-demand $/mo | 1yr RI no-upfront $/hr | 1yr $/mo | 1yr discount | 3yr RI no-upfront $/hr | 3yr $/mo | 3yr discount |
|---------------|--------------------|---------------:|---------------:|-----------------------:|---------:|-------------:|-----------------------:|---------:|-------------:|
| c7i.4xlarge   | none (16 vCPU CPU) | $0.8148        | $595           | $0.5371                | $392     | −34.1%       | $0.3696                | $270     | −54.6%       |
| g4dn.xlarge   | 1× NVIDIA T4       | $0.6580        | $480           | $0.4490                | $328     | −31.8%       | $0.3200                | $234     | −51.4%       |
| inf1.xlarge   | 1× Inferentia1 (4 cores) | $0.2850  | $208           | $0.1800                | $131     | −36.8%       | $0.1370                | $100     | −51.9%       |
| inf1.2xlarge  | 1× Inferentia1 (4 cores) | $0.4530  | $331           | $0.2850                | $208     | −37.1%       | $0.2170                | $158     | −52.1%       |
| inf2.xlarge   | 1× Inferentia2 (2 cores) | $1.1373  | $830           | $0.7165                | $523     | −37.0%       | $0.5241                | $383     | −53.9%       |
| g6.xlarge     | 1× NVIDIA L4       | $1.0064        | $735           | $0.6552                | $478     | −34.9%       | $0.4619                | $337     | −54.1%       |
| g5.xlarge     | 1× NVIDIA A10G     | $1.2580        | $918           | $0.7925                | $579     | −37.0%       | $0.5435                | $397     | −56.8%       |
| g6e.xlarge    | 1× NVIDIA L40S     | $2.3270        | $1,699         | $1.4660                | $1,070   | −37.0%       | $1.0053                | $734     | −56.8%       |
| g6.12xlarge   | 4× NVIDIA L4       | $5.7543        | $4,201         | $3.7460                | $2,735   | −34.9%       | $2.6412                | $1,928   | −54.1%       |

## Notes

- **Source**: AWS Price List Bulk API CSV for EC2 in `eu-central-1` — `https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/eu-central-1/index.csv` (the marketing pages at `aws.amazon.com/ec2/pricing/on-demand/` and `.../reserved-instances/pricing/` proxy this same data via a JS region selector, which `WebFetch` cannot drive — so I went directly to the authoritative bulk file).
- **Filter applied per row**: `TermType ∈ {OnDemand, Reserved}`, `Operating System = Linux`, `Tenancy = Shared`, `License Model = No License required`, `operation = RunInstances` (no Windows/RHEL/SQL surcharge), `CapacityStatus = Used`, and for RIs `OfferingClass = standard`, `PurchaseOption = No Upfront`, `LeaseContractLength ∈ {1yr, 3yr}`.
- **Inferentia RI availability**: `inf2.xlarge` *does* publish 1-yr and 3-yr Standard No-Upfront rates in `eu-central-1`, so it's filled in normally — no missing tiers for any of the seven instances at this price-list snapshot.
- **Discounts** are computed against the on-demand hourly rate from the same snapshot (negative numbers = savings).
- The "Convertible" RI class and "All / Partial Upfront" payment options also exist for every row above; they are not shown here per the request. As a rough reference, 3-yr Standard All-Upfront is typically a further ~6–10 % cheaper effective rate than the No-Upfront column shown.
- 730 h/mo is AWS's own monthly conversion convention (`365.25 × 24 / 12`). For a strict 30-day month, multiply hourly × 720.
- Prices exclude EBS, data transfer, Elastic IPs, and any Neuron/Driver-specific charges (none of these instances carry such per-hour add-ons in eu-central-1 today).
