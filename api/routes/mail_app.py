from fastapi_mail import FastMail, ConnectionConfig, MessageSchema, MessageType
from fastapi import APIRouter
from pydantic import BaseModel, EmailStr, Field, SecretStr
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()
mail_router = APIRouter()

class EmailDetails(BaseModel):
    sender_email: str
    sender_name: str
    sender_password: SecretStr
    recipients: List[EmailStr]
    tls: bool
    ssl: bool
    subject: str = Field(default="Lazi Invitation Awaiting!")
    template: str = Field(
        default="""
        <html>
            <body>
                <h1>{{ subject }}</h1>
                <p>Dear {{ recipient_name }},</p>
                <p>You have been invited to join Lazi.</p>
            </body>
        </html>
        """
    )
    subtype: MessageType = Field(default=MessageType.html)
    project_id: str | None = Field(default=None, description="The ID of the project from invite")
    organization_id: str | None = Field(default=None, description="The ID of the organization from invite")

class HandleMail:
    @staticmethod
    async def post_mail(email_details: EmailDetails):
        config = ConnectionConfig(
            MAIL_USERNAME=email_details.sender_email,
            MAIL_PASSWORD=email_details.sender_password,
            MAIL_FROM=email_details.sender_email,
            MAIL_PORT=587,
            MAIL_SERVER="smtp.gmail.com",
            MAIL_STARTTLS=email_details.tls,
            MAIL_SSL_TLS=email_details.ssl
        )
        message = MessageSchema(
            subject=email_details.subject,
            recipients=email_details.recipients,
            body=email_details.template,
            subtype=email_details.subtype
        )
        fm = FastMail(config)
        await fm.send_message(message)