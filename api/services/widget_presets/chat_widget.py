from api.config.fetchMongo import MongoHandler

project_config = MongoHandler("userAuthDatabase", "openProjects")

class ChatWidget:
    def __init__(self):
        self.w_client = WriteWidget(
            username="TestUser",
            token="TestToken",
            project_id="TestProjectID"
        )

    async def widget_base(self):
        self.chat_widget = await self.w_client.create(
            name="Chat Widget",
            size="large"
        )
