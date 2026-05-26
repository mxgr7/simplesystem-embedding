"""Pulumi program: stand up a TEI server EC2 + co-located load-gen EC2.

One stack per instance type (bench.sh creates them: bench-g6xlarge,
bench-c7i4xlarge, etc.). Config keys:

  aws:region                — defaults to eu-central-1
  bench:instanceType        — e.g. g6.xlarge / c7i.4xlarge / inf2.xlarge
  bench:hfToken             — passed through to TEI for model pull
  bench:publicKeyMaterial   — SSH public key (base64-armored OpenSSH);
                              private key stays on the operator's machine.

Resources per stack:
  - throwaway key pair (uploaded from config, not generated here, so
    the bench wrapper owns key lifecycle)
  - security group: SSH from anywhere, port 3000 only from the load-gen SG
  - server EC2 (the GPU/CPU/Inf box)
  - load-gen EC2 (always c7i.large, same AZ)

Outputs:
  - server_private_ip   (load-gen reaches TEI via this)
  - loadgen_public_ip   (operator SSHes here from their workstation)
  - server_public_ip    (for direct SSH debugging)
"""

from __future__ import annotations

from pathlib import Path

import pulumi
import pulumi_aws as aws

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
config = pulumi.Config("bench")
instance_type: str = config.require("instanceType")
# Not marked secret: the token ends up in user-data and is fetched in
# plaintext from the EC2 metadata service anyway. Marking it secret in
# Pulumi only encrypts the state file; it doesn't protect the runtime.
hf_token: str = config.require("hfToken")
public_key_material: str = config.require("publicKeyMaterial")

# Per-instance metadata: serving stack + TEI image tag.
# `kind` chooses AMI, user-data template, etc:
#   cpu  / gpu  = TEI Docker image (tag varies by NVIDIA arch)
#   inf1 / inf2 = bespoke neuron_server.py, multi-core via NEURON_RT_VISIBLE_CORES
# `cores` is the NeuronCore count per chip — userdata templatizes the systemd
# unit count + nginx upstream from this.
_INSTANCE_TABLE: dict[str, dict[str, object]] = {
    "c7i.4xlarge":  {"kind": "cpu",  "tei_tag": "cpu-1.9"},
    "g4dn.xlarge":  {"kind": "gpu",  "tei_tag": "turing-1.9"},
    "g4dn.2xlarge": {"kind": "gpu",  "tei_tag": "turing-1.9"},
    "g5.xlarge":    {"kind": "gpu",  "tei_tag": "86-1.9"},
    "g5.2xlarge":   {"kind": "gpu",  "tei_tag": "86-1.9"},
    "g6.xlarge":    {"kind": "gpu",  "tei_tag": "89-1.9"},
    "g6.2xlarge":   {"kind": "gpu",  "tei_tag": "89-1.9"},
    "g6.12xlarge":  {"kind": "gpu",  "tei_tag": "89-1.9"},
    "g6e.xlarge":   {"kind": "gpu",  "tei_tag": "89-1.9"},
    "g6e.2xlarge":  {"kind": "gpu",  "tei_tag": "89-1.9"},
    "inf1.xlarge":  {"kind": "inf1", "tei_tag": "n/a", "cores": 4},
    "inf1.2xlarge": {"kind": "inf1", "tei_tag": "n/a", "cores": 4},
    "inf2.xlarge":  {"kind": "inf2", "tei_tag": "n/a", "cores": 2},
}
if instance_type not in _INSTANCE_TABLE:
    raise ValueError(
        f"unknown instance type {instance_type!r}; "
        f"add it to _INSTANCE_TABLE in __main__.py"
    )
meta = _INSTANCE_TABLE[instance_type]
kind = str(meta["kind"])
tei_tag = str(meta["tei_tag"])
# inf1/inf2 need a few more knobs piped into the userdata template
cores = int(meta.get("cores", 0))

# --------------------------------------------------------------------------- #
# AMI selection
# --------------------------------------------------------------------------- #
# GPU + CPU: AWS Deep Learning Base AMI (Ubuntu 22.04). NVIDIA drivers,
# Docker, and nvidia-container-toolkit pre-installed. Use the SSM parameter
# rather than hard-coding an AMI ID so we always get the latest patched
# version.
DLAMI_GPU_SSM = (
    "/aws/service/deeplearning/ami/x86_64/"
    "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
)
# Inferentia2: AWS Deep Learning AMI Neuron multi-framework (Ubuntu 24.04).
# Ships /opt/aws_neuronx_venv_pytorch_2_9 (torch_neuronx for inf2/trn1).
# Notably does NOT include the inf1 venv — see DLAMI_NEURON1_SSM below.
DLAMI_NEURON2_SSM = (
    "/aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id"
)
# Inferentia1: dedicated single-purpose DLAMI on PyTorch 1.13 / Ubuntu 22.04.
# Inf1 SDK is frozen at PyTorch 1.13; the multi-framework DLAMI dropped it.
# Ships /opt/aws_neuron_venv_pytorch_* (torch_neuron, no x).
DLAMI_NEURON1_SSM = (
    "/aws/service/neuron/dlami/pt_1_13_inf1_ami_u22/ubuntu-22.04/latest/image_id"
)
# CPU only: plain Ubuntu 22.04 LTS — TEI's CPU image installs everything it
# needs in-container, so we don't need the DL base.
UBUNTU_2204_SSM = (
    "/aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id"
)

if kind == "inf2":
    server_ami_ssm = DLAMI_NEURON2_SSM
elif kind == "inf1":
    server_ami_ssm = DLAMI_NEURON1_SSM
elif kind == "gpu":
    server_ami_ssm = DLAMI_GPU_SSM
else:
    server_ami_ssm = UBUNTU_2204_SSM

server_ami = aws.ssm.get_parameter(name=server_ami_ssm).value
loadgen_ami = aws.ssm.get_parameter(name=UBUNTU_2204_SSM).value

# --------------------------------------------------------------------------- #
# Network — reuse the default VPC. We don't need anything fancier; saves
# ~30s of VPC creation per stack and a NAT gateway.
# --------------------------------------------------------------------------- #
default_vpc = aws.ec2.get_vpc(default=True)
default_subnets = aws.ec2.get_subnets(
    filters=[aws.ec2.GetSubnetsFilterArgs(name="vpc-id", values=[default_vpc.id])]
)
subnet_id = default_subnets.ids[0]
# Pin the load-gen and server to the same AZ to keep latency low.
subnet = aws.ec2.get_subnet(id=subnet_id)
az = subnet.availability_zone

# --------------------------------------------------------------------------- #
# Key pair — uploaded from config (private key lives on operator's box).
# --------------------------------------------------------------------------- #
key = aws.ec2.KeyPair(
    "bench-key",
    key_name=f"bench-{pulumi.get_stack()}",
    public_key=public_key_material,
)

# --------------------------------------------------------------------------- #
# Security groups
# --------------------------------------------------------------------------- #
# Load-gen box: SSH from anywhere (operator's laptop), egress unrestricted.
loadgen_sg = aws.ec2.SecurityGroup(
    "loadgen-sg",
    vpc_id=default_vpc.id,
    description="bench load-gen: ssh in, all egress",
    ingress=[
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp", from_port=22, to_port=22, cidr_blocks=["0.0.0.0/0"]
        ),
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            protocol="-1", from_port=0, to_port=0, cidr_blocks=["0.0.0.0/0"]
        ),
    ],
)

# Server box: SSH from anywhere (debugging), :3000 from load-gen SG only,
# egress unrestricted (HF model pull).
server_sg = aws.ec2.SecurityGroup(
    "server-sg",
    vpc_id=default_vpc.id,
    description="bench tei server: ssh + :3000 from loadgen",
    ingress=[
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp", from_port=22, to_port=22, cidr_blocks=["0.0.0.0/0"]
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp", from_port=3000, to_port=3000,
            security_groups=[loadgen_sg.id],
        ),
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            protocol="-1", from_port=0, to_port=0, cidr_blocks=["0.0.0.0/0"]
        ),
    ],
)

# --------------------------------------------------------------------------- #
# Server user-data
# --------------------------------------------------------------------------- #
HERE = Path(__file__).parent

if kind in ("inf1", "inf2"):
    # neuron_server.py + model files arrive via SCP from bench.sh (same
    # pattern as loadgen.py). hf_token unused on this path — model
    # artifacts come from /tei-models on the operator box.
    # Substitute the venv glob + NeuronCore count so the same template
    # serves both inf1 (4 cores, no-x venv) and inf2 (2 cores, x venv).
    if kind == "inf1":
        neuron_venv_glob = "/opt/aws_neuron_venv_pytorch_*"
    else:
        neuron_venv_glob = "/opt/aws_neuronx_venv_pytorch_*"
    tmpl = (HERE / "inferentia_userdata.sh").read_text()
    server_user_data = (
        tmpl
        .replace("__NEURON_VENV_GLOB__", neuron_venv_glob)
        .replace("__NUM_CORES__", str(cores))
    )
else:
    tmpl = (HERE / "tei_userdata.sh").read_text()
    server_user_data = (
        tmpl
        .replace("__HF_TOKEN__", hf_token)
        .replace("__TEI_IMAGE_TAG__", tei_tag)
        .replace("__INSTANCE_KIND__", kind)
    )

# --------------------------------------------------------------------------- #
# Load-gen user-data — minimal: just python + httpx, plus copy loadgen.py.
# --------------------------------------------------------------------------- #
# loadgen.py is NOT embedded — base64 of the source pushes user-data past
# AWS's 16,384-byte limit. bench.sh SCPs it after boot instead.
loadgen_user_data = """#!/usr/bin/env bash
set -uo pipefail
exec > >(tee -a /var/log/loadgen-userdata.log) 2>&1
log() { printf '[%s | loadgen-userdata] %s\\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
log "installing python3 + curl"
apt-get update -q
DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3 curl ca-certificates
log "installing uv"
curl -fsSL https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
log "uv version: $(uv --version)"
log "installing httpx + numpy with uv (system python)"
uv pip install --system --quiet httpx numpy
mkdir -p /opt/loadgen
chown ubuntu:ubuntu /opt/loadgen
log "deps installed; awaiting loadgen.py scp'd in by orchestrator"
echo "ok" > /var/run/loadgen-ready
"""

# --------------------------------------------------------------------------- #
# Disk sizing — Deep Learning AMI itself uses ~50 GB; TEI model + Docker
# layer ~3 GB; HF cache ~1.5 GB. 80 GB is comfortable for GPU, 40 GB is
# plenty for the plain CPU box. The Neuron multi-framework DLAMI is much
# fatter — its EBS snapshot is 100 GB, so the root volume cannot be smaller
# than that (RunInstances 400 InvalidBlockDeviceMapping otherwise).
# --------------------------------------------------------------------------- #
if kind == "cpu":
    root_size_gb = 40
elif kind == "inf":
    root_size_gb = 120
else:
    root_size_gb = 80

server = aws.ec2.Instance(
    "tei-server",
    instance_type=instance_type,
    ami=server_ami,
    subnet_id=subnet_id,
    vpc_security_group_ids=[server_sg.id],
    key_name=key.key_name,
    user_data=server_user_data,
    associate_public_ip_address=True,
    root_block_device=aws.ec2.InstanceRootBlockDeviceArgs(
        volume_size=root_size_gb,
        volume_type="gp3",
        delete_on_termination=True,
    ),
    tags={"Name": f"bench-tei-{instance_type}", "Project": "aws-embedding-bench"},
)

loadgen = aws.ec2.Instance(
    "loadgen",
    instance_type="c7i.large",
    ami=loadgen_ami,
    subnet_id=subnet_id,
    vpc_security_group_ids=[loadgen_sg.id],
    key_name=key.key_name,
    user_data=loadgen_user_data,
    associate_public_ip_address=True,
    root_block_device=aws.ec2.InstanceRootBlockDeviceArgs(
        volume_size=20,
        volume_type="gp3",
        delete_on_termination=True,
    ),
    tags={"Name": f"bench-loadgen-{instance_type}", "Project": "aws-embedding-bench"},
)

pulumi.export("server_private_ip", server.private_ip)
pulumi.export("server_public_ip", server.public_ip)
pulumi.export("loadgen_public_ip", loadgen.public_ip)
pulumi.export("instance_type", instance_type)
pulumi.export("availability_zone", az)
