import smtplib, ssl

password = input("Type your password and press enter: ")
port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "Larrain Vial Quant Team"  # Enter your address
receiver_email = "fpaniagua@larrainvial.com"  # Enter receiver address
message = """\
Subject: Hi there

This message is sent from Python."""

# Create a secure SSL context
context = ssl.create_default_context()

with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
    server.login("lv.quant.team@gmail.com", password)
    server.sendmail(sender_email, receiver_email, message)