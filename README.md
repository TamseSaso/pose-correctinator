

# Pose Correctinator

A real-time posture-correction example for the Luxonis OAK-4. Runs the neural-net on the camera and alerts your laptop when posture is incorrect.

## Prerequisites

- An OAK-4 camera with network access (PoE or Ethernet)  
- Python 3.7+ on both camera and laptop  
- On the **camera**, install:
  ```bash
  pip install depthai depthai-nodes numpy
  ```
- On the **laptop**, install:
  ```bash
  pip install simpleaudio
  ```
- `oakctl` CLI installed and configured

## Setup & Usage

1. **Clone this repository**  
   ```bash
   git clone https://github.com/TamseSaso/pose-correctinator.git
   cd pose-correctinator
   ```

2. **Configure your laptop IP**  
   Open `main.py` and update the `LAPTOP_IP` constant to your laptop’s local network address:
   ```python
   LAPTOP_IP = "192.168.1.100"  # replace with your laptop’s IP
   ```

3. **Start the listener on your laptop**  
   In one terminal, run:
   ```bash
   python posture_alert_listener.py
   ```

4. **Deploy & run on the OAK-4**  
   In another terminal (still in the repo root), run:
   ```bash
   oakctl app run .
   ```

Now, whenever the OAK-4 detects incorrect posture, it will send a UDP ping to your laptop and you’ll hear the alert sound.  