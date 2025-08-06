- USB Ethernet Gadget (Pi Zero as a USB NIC)**
    - The Pi Zero can act as a **USB Ethernet device** (`usb0`) to use SSH or TCP with your Python app over USB.
- Base Configuration (/boot)
    - Add to `/boot/config.txt`（append):
    - Append to `/boot/cmdline.txt`：
        ```
        modules_load=dwc2,g_ether
        ```
    - Enable SSH (create an empty `ssh` file in `/boot`)：
        ```bash
        sudo touch /boot/ssh
        ```

        ```ini
        dtoverlay=dwc2
        ```
- Connect and Get IP (Windows)
    - Connect the Pi to the PC via USB.
    - Check network adapters：
        ```powershell
        ipconfig
        ```
        - Expect to see **Ethernet adapter Ethernet** (the Pi’s USB NIC).
    - If “Connection refused” appears：
        - Check and fix ICS (Internet Connection Sharing) service.
- Network Checks on the Pi
    - Inspect the USB NIC：
        ```bash
        ifconfig usb0
        ```
    - Determine whether SSH is via Wi-Fi or USB：
        ```bash
        set -- $SSH_CONNECTION
        ip route get "$1"
        ```
- FastAPI Deployment and Access (Pi)
    - Install：
        ```bash
        sudo apt update
        sudo apt install -y python3-fastapi python3-uvicorn
        ```
    - Start your app (example)：
		```bash
		python3 dmd_fastapi.py
		```
        ```bash
        uvicorn dmd_fastapi:app --host 0.0.0.0 --port 8000
        ```
    - Access in browser (typical USB NIC address)：
        - `http://192.168.137.2:8000/docs`
- **File Sync and Account**
    - WinSCP sync directory: `uc2dmd`
    - Account:
        - Username: `dmd2`
        - Password: `123` (internal/test only)
- **Common IPs and Access**
    - USB direct (g_ether):
        - `http://192.168.137.2:8000/docs`
    - Router/non-USB path (LAN):
        - `http://192.168.178.131:8000/docs` (not the USB link)
- Quick Commands
    - Windows：
        ```bash
        ipconfig
        ```
    - Pi：
        ```bash
        ifconfig usb0
        set -- $SSH_CONNECTION
        ip route get "$1"
        ```
