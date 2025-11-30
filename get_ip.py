"""
IP Detection Utility
Automatically detects local IP address
"""

import socket

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Create a socket connection to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to an external address (doesn't actually send data)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        print(f"Could not detect IP: {e}")
        return "127.0.0.1"  # Fallback to localhost

if __name__ == "__main__":
    ip = get_local_ip()
    print(f"Your local IP: {ip}")
