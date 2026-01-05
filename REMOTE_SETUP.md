# Remote Node Setup - Quick Start

## Prerequisites

1. **SSH key-based authentication** to remote machines
2. **Python 3.9+** and **thinkagain[grpc]** installed on all machines
3. **Network connectivity** between all nodes

## Setup Steps

### 1. On Remote Nodes

```bash
# Install dependencies
pip3 install 'thinkagain[grpc]'

# Or with virtual environment (recommended)
python3 -m venv /opt/thinkagain-env
source /opt/thinkagain-env/bin/activate
pip install 'thinkagain[grpc]'
```

### 2. On Client Machine

```bash
# Setup SSH keys
ssh-keygen -t rsa -f ~/.ssh/thinkagain_key
ssh-copy-id -i ~/.ssh/thinkagain_key.pub user@worker1.example.com

# Test connection (should not ask for password)
ssh -i ~/.ssh/thinkagain_key user@worker1.example.com
```

### 3. Configure and Deploy

```python
from thinkagain.distributed import init, replica
from thinkagain.distributed.nodes import NodeConfig

# Configure cluster
nodes = [
    # Local node
    NodeConfig(host="localhost", cpus=8, gpus=0),

    # Remote node
    NodeConfig(
        host="worker1.example.com",
        cpus=16,
        gpus=2,
        ssh_user="ubuntu",
        ssh_key_path="~/.ssh/thinkagain_key",
        # If using venv on remote:
        env_setup="source /opt/thinkagain-env/bin/activate",
    ),
]

# Initialize
init(backend="grpc", nodes=nodes)

# Deploy
@replica(cpus=2)
class MyService:
    def work(self):
        return "done"

await MyService.deploy(instances=3)
```

## Common Issues

**SSH permission denied:**
```bash
chmod 600 ~/.ssh/thinkagain_key
```

**Remote server fails to start:**
```bash
# SSH to remote and verify:
python3 -c "import thinkagain; print('OK')"
```

**Can't connect to remote server:**
- Check firewall allows gRPC ports (50000-60000)
- Verify network connectivity between nodes

## See Also

- [examples/remote_deployment_demo.py](examples/remote_deployment_demo.py) - Full example
- [PHASE1_IMPLEMENTATION.md](PHASE1_IMPLEMENTATION.md) - Architecture details
