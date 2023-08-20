import pywhatkit
BASE_ISRAEL = "+972"
NOYA = "538279402"
for i in range(100):
    pywhatkit.sendwhatmsg_instantly(phone_no=f"{BASE_ISRAEL}{NOYA}", message="מה את עושה נויה?", wait_time=4)
