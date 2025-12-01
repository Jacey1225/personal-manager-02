from api.services.lazi_sdk.widgets_write import WriteWidget
from api.schemas.widgets import WidgetSize
import asyncio
import os
from typing import Optional

class DiscussionWidget:
    """
    Creates a discussion board widget for team communication.
    Uses the public API to build an interactive message board.
    """
    
    def __init__(self, username: str, token: str, project_id: str):
        self.w_client = WriteWidget(
            username=username,
            token=token,
            project_id=project_id
        )
    
    async def create_discussion_board(self, board_name: str = "Team Discussion"):
        """
        Create a complete discussion board widget with message input and send functionality.
        
        Args:
            board_name: Name of the discussion board
        """
        
        # Step 1: Create the base widget
        await self.w_client.create(
            name=board_name,
            size=WidgetSize.LARGE.value
        )
        
        # Step 2: Define the interaction endpoint and logic
        await self.w_client.interaction(
            endpoint="/api/v1/widgets/discussion/post_message",
            headers={
                "Content-Type": "application/json",
                "X-WebSocket-Enabled": "true"  # Enable WebSocket for real-time updates
            },
            refresh_interval=10,  # Fallback: Refresh every 10 seconds if WebSocket disconnects
            func=self._message_handler
        )
        
        # Step 3: Add components for the discussion board
        
        # Header container
        await self.w_client.component(
            endpoint="/api/v1/widgets/discussion/post_message",
            type="container",
            content=[],
            props={
                "id": "header_container",
                "position": "relative",
                "top": 0,
                "left": 0,
                "width": "100%",
                "height": "60px",
                "background": "#f8f9fa",
                "border_bottom": "2px solid #dee2e6",
                "padding": {"top": 16, "right": 20, "bottom": 16, "left": 20},
                "display": "flex",
                "align_items": "center",
                "justify_content": "space-between"
            }
        )
        
        # Board title
        await self.w_client.component(
            endpoint="/api/v1/widgets/discussion/post_message",
            type="text",
            content=[],
            props={
                "id": "board_title",
                "content": board_name,
                "font_size": 20,
                "font_weight": "bold",
                "color": "#212529",
                "margin": 0
            }
        )
        
        # Message count badge
        await self.w_client.component(
            endpoint="/api/v1/widgets/discussion/post_message",
            type="badge",
            content=[],
            props={
                "id": "message_count",
                "label": "{{total_messages}} messages",
                "background": "#0d6efd",
                "color": "#ffffff",
                "padding": {"top": 6, "right": 12, "bottom": 6, "left": 12},
                "border_radius": 12,
                "font_size": 13,
                "font_weight": "600"
            }
        )
        
        # Messages container (scrollable area)
        # Note: The content field will store message dictionaries
        await self.w_client.component(
            endpoint="/api/v1/widgets/discussion/post_message",
            type="message_list",
            content=[],  # This will be populated with message dictionaries
            props={
                "id": "messages_container",
                "position": "relative",
                "width": "100%",
                "height": "calc(100% - 140px)",
                "top": 60,
                "background": "#ffffff",
                "padding": {"top": 20, "right": 20, "bottom": 20, "left": 20},
                "overflow_y": "auto",
                "overflow_x": "hidden",
                "render_type": "list",
                "item_template": "message_item"
            }
        )
        
        # Message item template (how each message displays)
        await self.w_client.component(
            endpoint="/api/v1/widgets/discussion/post_message",
            type="message_item",
            content=[],
            props={
                "id": "message_item",
                "is_template": True,
                "display": "flex",
                "flex_direction": "column",
                "margin_bottom": 16,
                "padding": 12,
                "background": "#f8f9fa",
                "border_radius": 8,
                "border_left": "4px solid #0d6efd",
                
                # Message header
                "username": "{{message.username}}",
                "timestamp": "{{message.timestamp}}",
                "username_font_size": 14,
                "username_font_weight": "600",
                "username_color": "#495057",
                "timestamp_font_size": 11,
                "timestamp_color": "#6c757d",
                
                # Message content
                "text": "{{message.text}}",
                "text_font_size": 14,
                "text_color": "#212529",
                "text_margin_top": 8,
                "text_line_height": 1.5,
                "text_word_wrap": "break-word"
            }
        )
        
        # Input area container
        await self.w_client.component(
            endpoint="/api/v1/widgets/discussion/post_message",
            type="container",
            content=[],
            props={
                "id": "input_container",
                "position": "absolute",
                "bottom": 0,
                "left": 0,
                "width": "100%",
                "height": "80px",
                "background": "#ffffff",
                "border_top": "2px solid #dee2e6",
                "padding": {"top": 16, "right": 20, "bottom": 16, "left": 20},
                "display": "flex",
                "align_items": "center",
                "gap": 12
            }
        )
        
        # Message input field
        await self.w_client.component(
            endpoint="/api/v1/widgets/discussion/post_message",
            type="input",
            content=[],
            props={
                "id": "message_input",
                "name": "message_text",
                "type": "text",
                
                # Size
                "flex": 1,
                "height": 48,
                
                # Styling
                "border": "2px solid #ced4da",
                "border_radius": 6,
                "padding": {"top": 12, "right": 16, "bottom": 12, "left": 16},
                "font_size": 14,
                "color": "#212529",
                "background": "#ffffff",
                
                # Behavior
                "placeholder": "Type your message here...",
                "max_length": 1000,
                "autocomplete": "off",
                "required": True,
                
                # Focus state
                "focus_border_color": "#0d6efd",
                "focus_outline": "none",
                
                # Events
                "on_enter": "submit_message"
            }
        )
        
        # Send button
        await self.w_client.component(
            endpoint="/api/v1/widgets/discussion/post_message",
            type="button",
            content=[],
            props={
                "id": "send_button",
                "label": "Send",
                
                # Size
                "width": 100,
                "height": 48,
                
                # Styling
                "background": "#0d6efd",
                "color": "#ffffff",
                "border": "none",
                "border_radius": 6,
                "font_size": 14,
                "font_weight": "600",
                "cursor": "pointer",
                
                # Hover state
                "hover_background": "#0b5ed7",
                "transition": "all 0.2s ease",
                
                # Behavior
                "submit_type": "post_message",
                "loading_text": "Sending...",
                "disabled_background": "#6c757d",
                
                # Events
                "on_click": "submit_message"
            }
        )
        
        # Step 4: Save the widget
        await self.w_client.save()
        
        return self.w_client.current_widget
    
    def _message_handler(self, widget_data, request, user):
        """
        Handler function for posting and retrieving messages.
        Messages are stored in the component's content field as dictionaries.
        This code will be executed in the sandboxed environment.
        """
        import datetime
        import uuid
        
        # Get the message text from the request
        message_text = request.params.get('message_text', '').strip()
        
        # Retrieve existing messages from the message_list component's content field
        # In the actual widget, this would access the component's content array
        existing_messages = widget_data.get('component_content', {}).get('message_list', [])
        
        # Validate message
        if message_text and len(message_text) > 0:
            if len(message_text) > 1000:
                return {
                    'status': 'error',
                    'message': 'Message too long (max 1000 characters)',
                    'component_updates': {
                        'message_list': {
                            'content': existing_messages
                        }
                    },
                    'data': {
                        'total_messages': len(existing_messages)
                    }
                }
            
            # Create new message dictionary to store in content field
            new_message = {
                'id': str(uuid.uuid4()),
                'user_id': user.id,
                'username': user.username,
                'text': message_text,
                'timestamp': datetime.datetime.now().isoformat(),
                'edited': False,
                'reactions': []
            }
            
            # Add new message to the content array
            existing_messages.append(new_message)
            
            # Return component updates to store messages in content field
            return {
                'status': 'success',
                'message': 'Message posted',
                'component_updates': {
                    'message_list': {
                        'content': existing_messages  # Update the content field with all messages
                    }
                },
                'data': {
                    'messages': existing_messages[-50:],  # Display last 50 messages
                    'total_messages': len(existing_messages),
                    'new_message': new_message
                },
                'broadcast': True,  # Signal to broadcast this update to all connected clients via WebSocket
                'broadcast_event': 'message_posted',
                'broadcast_data': {
                    'message': new_message,
                    'total_count': len(existing_messages)
                }
            }
        else:
            # Just fetch messages without posting
            return {
                'status': 'success',
                'component_updates': {
                    'message_list': {
                        'content': existing_messages
                    }
                },
                'data': {
                    'messages': existing_messages[-50:],  # Display last 50 messages
                    'total_messages': len(existing_messages)
                }
            }


# Example usage function
async def create_project_discussion(username: str, token: str, project_id: str):
    """
    Helper function to create a discussion board for a project.
    
    Usage:
        widget = await create_project_discussion(
            username="user@example.com",
            token="your_oauth_token",
            project_id="proj_123abc"
        )
    """
    discussion = DiscussionWidget(username, token, project_id)
    widget = await discussion.create_discussion_board(board_name="Project Discussion")
    
    print(f"✅ Discussion board created!")
    print(f"   Widget ID: {widget.id}")
    print(f"   Name: {widget.name}")
    print(f"   Size: {widget.size}")
    print(f"   Endpoint: /api/v1/widgets/discussion/post_message")
    
    return widget


# Command-line test function
async def test_discussion_widget():
    """Test function to create a discussion widget"""
    
    # Replace with actual credentials
    TEST_USERNAME = os.getenv("TEST_USERNAME", "test@example.com")
    TEST_TOKEN = os.getenv("TEST_TOKEN", "test_token")
    TEST_PROJECT_ID = os.getenv("TEST_PROJECT_ID", "test_project")
    
    try:
        widget = await create_project_discussion(
            username=TEST_USERNAME,
            token=TEST_TOKEN,
            project_id=TEST_PROJECT_ID
        )
        print("\n🎉 Discussion widget created successfully!")
        return widget
    except Exception as e:
        print(f"\n❌ Error creating discussion widget: {e}")
        raise


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_discussion_widget())
