# Intelligent RL Resource Allocator

An advanced Reinforcement Learning (RL) agent based on Proximal Policy Optimization (PPO) designed to dynamically manage and optimize Docker container resources. The agent monitors real-time system metrics and autonomously adjusts CPU shares and memory limits to balance application performance with infrastructure cost efficiency.

## 🚀 Live Deployment
You can interact with the live dashboard and see the agent in action here:
**[View Live Dashboard on Hugging Face](https://huggingface.co/spaces/prat23/resource-allocator)**

## 🌟 Key Features
- **Real-time Monitoring**: Connects directly to the Docker engine via `docker.sock` to fetch live container stats.
- **Dynamic Optimization**: The PPO agent learns workload patterns and proactively adjusts resource allocations before performance bottlenecks occur.
- **Cost Reduction**: Minimizes over-provisioning by scaling down under-utilized containers while ensuring zero downtime.
- **Intuitive Dashboard**: Built with Streamlit to visualize multi-container metrics, agent rewards, and cost-savings analytics.

## 🛠️ System Architecture
The system consists of three main components:
1. **Core Environment**: A custom Gymnasium-compliant environment that models Docker container states.
2. **FastAPI Backend**: A high-performance API that serves the model and handles environment resets/steps.
3. **Streamlit Frontend**: A professional-grade dashboard for real-time visualization and performance monitoring.

## 📋 Action & Observation Spaces

### Observation Space
The agent perceives the system state through a `ResourceObservation` model:
- **Container Metrics**: CPU percentage, memory usage, current limits, and usage trends.
- **System Metrics**: Total CPU availability, available memory (MB), and system load average.
- **Workload Priority**: Integrated priority levels (Critical, High, Normal, Low) to influence allocation decisions.

### Action Space
The agent can execute the following adjustments:
- **CPU Tuning**: Adjust `cpu_shares` in fine-grained steps (-256 to +256).
- **Memory Scaling**: Dynamically modify memory limits based on safety thresholds.
- **Priority Management**: Shift container priorities based on real-time importance.

## 💻 Local Installation & Usage

To run this project on any desktop or server environment, follow these steps:

### Prerequisites
- Docker (with the Docker socket accessible)
- Python 3.10+
- `pip` or `uv`

### Step 1: Clone the Repository
```bash
git clone https://github.com/prateeksha-khichi/Resource-allocator.git
cd Resource-allocator
```

### Step 2: Running with Docker (Recommended)
This ensures all dependencies and binary libraries are handled correctly.
```bash
docker build -t rl-resource-allocator .
docker run -v /var/run/docker.sock:/var/run/docker.sock -p 8000:8000 -p 7860:7860 rl-resource-allocator
```
- API Endpoint: `http://localhost:8000`
- Dashboard: `http://localhost:7860`

### Step 3: Manual Installation (Development)
```bash
pip install -r requirements.txt
python -m server.app  # Start the API
streamlit run dashboard.py # Start the Dashboard
```

## 📊 Performance Benchmarks
The agent is evaluated across three primary scenarios:
1. **Stability**: Maintaining optimal resource usage for stable workloads.
2. **Prioritization**: Ensuring critical services receive resources during high system load.
3. **Recovery**: Rapidly allocating resources during sudden workload spikes.

---
Built with ❤️ by [Prateeksha Khichi](https://github.com/prateeksha-khichi)
