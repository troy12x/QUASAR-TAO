#!/bin/bash
# Script to open port 8091 for the miner's axon connection
# This port is used by Bittensor for peer-to-peer communication

echo "=========================================="
echo "Opening Port 8091 for Miner Axon"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  This script needs root privileges to modify firewall rules"
    echo "   Please run with: sudo ./open_port_8091.sh"
    exit 1
fi

# Method 1: UFW (Ubuntu/Debian)
if command -v ufw &> /dev/null; then
    echo "✅ Found UFW firewall"
    echo "   Opening port 8091 (TCP)..."
    ufw allow 8091/tcp
    echo "   Opening port 8091 (UDP)..."
    ufw allow 8091/udp
    echo "   Reloading firewall..."
    ufw reload
    echo "✅ Port 8091 opened with UFW"
    echo ""
fi

# Method 2: firewalld (CentOS/RHEL/Fedora)
if command -v firewall-cmd &> /dev/null; then
    echo "✅ Found firewalld"
    echo "   Opening port 8091 (TCP)..."
    firewall-cmd --permanent --add-port=8091/tcp
    echo "   Opening port 8091 (UDP)..."
    firewall-cmd --permanent --add-port=8091/udp
    echo "   Reloading firewall..."
    firewall-cmd --reload
    echo "✅ Port 8091 opened with firewalld"
    echo ""
fi

# Method 3: iptables (if no other firewall is detected)
if ! command -v ufw &> /dev/null && ! command -v firewall-cmd &> /dev/null; then
    echo "⚠️  No UFW or firewalld detected"
    echo "   Attempting to open port 8091 with iptables..."
    
    # Check if iptables is available
    if command -v iptables &> /dev/null; then
        # Allow TCP
        iptables -I INPUT -p tcp --dport 8091 -j ACCEPT
        # Allow UDP
        iptables -I INPUT -p udp --dport 8091 -j ACCEPT
        
        # Try to save rules (varies by distribution)
        if command -v iptables-save &> /dev/null; then
            iptables-save > /etc/iptables/rules.v4 2>/dev/null || \
            iptables-save > /etc/iptables.rules 2>/dev/null || \
            echo "⚠️  iptables rules added but may not persist after reboot"
        fi
        
        echo "✅ Port 8091 opened with iptables"
        echo "   Note: Rules may need to be saved manually depending on your system"
    else
        echo "❌ iptables not found"
    fi
    echo ""
fi

# Verify port is listening (if netstat/ss is available)
if command -v ss &> /dev/null; then
    echo "Checking if port 8091 is open..."
    ss -tuln | grep 8091 && echo "✅ Port 8091 appears to be open" || echo "⚠️  Port 8091 not yet listening (miner may not be running)"
elif command -v netstat &> /dev/null; then
    echo "Checking if port 8091 is open..."
    netstat -tuln | grep 8091 && echo "✅ Port 8091 appears to be open" || echo "⚠️  Port 8091 not yet listening (miner may not be running)"
fi

echo ""
echo "=========================================="
echo "Important Notes:"
echo "=========================================="
echo "1. If you're using a cloud provider (AWS, GCP, Azure, etc.),"
echo "   you may also need to open port 8091 in their security groups/firewall"
echo ""
echo "2. For AWS: Add inbound rule for port 8091 (TCP and UDP) in Security Groups"
echo "3. For GCP: Add firewall rule for port 8091 in VPC Firewall Rules"
echo "4. For Azure: Add inbound rule for port 8091 in Network Security Groups"
echo ""
echo "5. Verify your miner is configured to use port 8091:"
echo "   --axon.port 8091"
echo ""
echo "6. Test the port from outside:"
echo "   telnet YOUR_SERVER_IP 8091"
echo "   or"
echo "   nc -zv YOUR_SERVER_IP 8091"
echo "=========================================="
