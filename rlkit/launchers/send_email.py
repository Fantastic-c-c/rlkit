from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
import smtplib

def _send_email(message, start_str, attachment=None, subject=None):
    _email_credentials = {
            'address': 'lovebot360@gmail.com',
            'password': 'railrobots',
            'receivers': ['krakelly@gmail.com']
            }
    # loads credentials and receivers
    address, password = _email_credentials['address'], _email_credentials['password']
    if 'gmail' in address:
        smtp_server = "smtp.gmail.com"
    else:
        raise NotImplementedError
    receivers = _email_credentials['receivers']

    # configure default subject
    if subject is None:
        subject = 'Sawyer Robot Experiment Update: {} started on {}'.format('Laudri', start_str)

    # constructs message
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = address
    msg['To'] = ', '.join(receivers)
    msg.attach(MIMEText(message))

    if attachment:
        attached_part = MIMEBase('application', "octet-stream")
        attached_part.set_payload(open(attachment, "rb").read())
        Encoders.encode_base64(attached_part)
        attached_part.add_header('Content-Disposition', 'attachment; filename="{}"'.format(attachment))
        msg.attach(attached_part)

    # logs in and sends
    server = smtplib.SMTP_SSL(smtp_server)
    server.login(address, password)
    server.sendmail(address, receivers, msg.as_string())

