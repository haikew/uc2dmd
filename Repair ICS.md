**Sharing Internet to Raspberry Pi over USB (RNDIS) with Static IP, fixing ICS, proving SSH runs over USB (not Wi-Fi)**  
_Environment: Windows 11 Pro host + Raspberry Pi (`usb0`). No WinNAT. Static IP only._

---

### 1) Repair Windows components (if ICS is missing/broken) — run **as Administrator** (PowerShell)

```powershell
DISM /Online /Cleanup-Image /CheckHealth
DISM /Online /Cleanup-Image /ScanHealth
DISM /Online /Cleanup-Image /RestoreHealth
sfc /scannow
```

Verify ICS service:

```powershell
sc.exe qc sharedaccess
sc.exe queryex sharedaccess
```

---

### 2) Enable ICS (Wi-Fi → USB/RNDIS)

- `ncpa.cpl` → Wi-Fi → **Properties → Sharing** → enable _Allow other network users…_ → select **USB Ethernet/RNDIS Gadget**.
    
- Expected: RNDIS becomes **192.168.137.1/24**. (https://learn.microsoft.com/en-us/answers/questions/2699518/to-change-default-internet-connection-sharing-ip-a?utm_source=chatgpt.com "To Change Default Internet Connection Sharing IP Address ...")
    

Quick checks:

```cmd
ipconfig /all
sc queryex sharedaccess
```

---

### 3) Configure Raspberry Pi for **USB Ethernet Gadget**

Edit SD card (or over SSH then reboot):

**/boot/config.txt**

```ini
dtoverlay=dwc2
```

**/boot/cmdline.txt** (append on the single line)

```text
modules_load=dwc2,g_ether
```

Enable SSH:

```bash
sudo touch /boot/ssh
```

Connect Pi to the PC via USB.

---

### 4) Set **Static IP** on Pi (`usb0`)

```bash
sudo nano /etc/dhcpcd.conf
```

Append:

```ini
interface usb0
  static ip_address=192.168.137.2/24
  static routers=192.168.137.1
  static domain_name_servers=1.1.1.1 8.8.8.8
```

Apply:

```bash
sudo systemctl restart dhcpcd.service
ip addr show usb0
ping -c3 192.168.137.1
```

---

### 4.5) If Windows does **not** recognize the Pi as a network adapter (needs driver)

1. Open Device Manager (`devmgmt.msc`).
    
2. Find the Pi device (often under **Other devices**, sometimes shown as _RNDIS_, _USB Ethernet/RNDIS Gadget_, or a COM device).
    
3. **Update driver** → **Browse my computer** → **Let me pick from a list** → **Network adapters** → **Microsoft** → **Remote NDIS Compatible Device** → **Next**. (https://learn.microsoft.com/en-us/answers/questions/3225865/missing-remote-ndis-driver-after-win10-update-to-1?utm_source=chatgpt.com "Missing Remote NDIS DRIVER after win10 update to 1903"), [learn.adafruit.com](https://learn.adafruit.com/turning-your-raspberry-pi-zero-into-a-usb-gadget/ethernet-gadget?utm_source=chatgpt.com "Turning your Raspberry Pi Zero into a USB Gadget")
    
4. If it still isn’t listed: **Have Disk…** → path **`C:\Windows\INF`** → pick **netrndis.inf** (or select Microsoft → RNDIS model) → **Next**. (https://learn.microsoft.com/en-us/answers/questions/2790061/remote-ndis-based-internet-sharing-device-no-valid?utm_source=chatgpt.com "Remote NDIS based internet sharing Device \"No valid IP ...")
    
5. Background: RNDIS is a built-in Windows class driver; binding it exposes the Pi as an Ethernet NIC over USB. (https://learn.microsoft.com/en-us/windows-hardware/drivers/network/remote-ndis--rndis-2?utm_source=chatgpt.com "Introduction to Remote NDIS (RNDIS) - Windows drivers"), (https://learn.microsoft.com/en-us/windows-hardware/drivers/network/overview-of-remote-ndis--rndis-?utm_source=chatgpt.com "Overview of Remote NDIS (RNDIS) - Windows drivers")
    

---

### 5) Find IPs (optional)

Windows:

```cmd
ipconfig
```

Pi:

```bash
ifconfig usb0
```

---

### 6) SSH (force USB path)

Windows:

```cmd
ssh dmd2@192.168.137.2
```

If “Connection refused”:

```bash
# on Pi
sudo systemctl status ssh
sudo systemctl enable --now ssh
```

---

### 7) Prove SSH uses **USB**, not Wi-Fi

**On Windows:**

```cmd
route print 192.168.137.0
```

```powershell
Get-NetNeighbor -IPAddress 192.168.137.2 | Format-List ifIndex,InterfaceAlias,LinkLayerAddress,State
```

(InterfaceAlias should be the RNDIS adapter.)

**On Pi:**

```bash
ip route get 192.168.137.1
ss -tuna | grep ':22'
sudo ip link set wlan0 down    # optional hard proof
# or: sudo rfkill block wifi
```

---

### 8) Quick connectivity tests

**Windows:**

```cmd
ping 192.168.137.2 -n 3
ping 1.1.1.1 -n 3
ping google.com -n 3
```

**Pi:**

```bash
ping -c3 192.168.137.1
ping -c3 1.1.1.1
ping -c3 google.com
```

---

### 9) WinSCP file sync (optional)

- Host: `192.168.137.2`
    
- User: `dmd2`
    
- Password: `123`
    
- Sync folder: `uc2dmd`
    

---

### 10) Run your FastAPI app on Pi

```bash
sudo apt update
sudo apt install -y python3-fastapi python3-uvicorn
uvicorn dmd_fastapi:app --host 0.0.0.0 --port 8000
```

Access:

- **USB (wired)**: `http://192.168.137.2:8000/docs`
    
- **Wi-Fi (for comparison)**: `http://192.168.178.131:8000/docs`
    

---

### Minimal regression checklist

```cmd
:: Windows
sc queryex sharedaccess
ipconfig | findstr /i "RNDIS 192.168.137.1"
ping 192.168.137.2 -n 3
ssh dmd2@192.168.137.2
```

```bash
# Pi
ip addr show usb0
ip route get 192.168.137.1
ss -tuna | grep ':22'
```
