from api.schemas.calendar import CalendarInsights, CalendarEvent, DateTimeSet
from api.schemas.model import InputRequest, EventOutput
from api.services.model_setup.structure_model_output import HandleResponse
from api.services.calendar.handleEvents import AddToCalendar, DeleteFromCalendar, UpdateFromCalendar
from api.services.calendar.eventSetup import RequestSetup
from api.config.uniformInterface import UniformInterface
from api.config.fetchMongo import MongoHandler
from typing import Optional, Dict, Any
from datetime import datetime
import pytz
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

responseRequest = {
    "user_id": str,
    "status": str,
    "message": str,
    "event_requested": dict,
    "calendar_insights": Optional[dict]
}

user_config = MongoHandler(None, "userAuthDatabase", "userCredentials")
project_config = MongoHandler(None, "userAuthDatabase", "openProjects")

class MainModel:
    @staticmethod
    def format_response(status: str, message: str, event_requested: dict, calendar_insights: dict = {}) -> dict:
        """Processes the response for the user request.

        Args:
            status (str): The status of the response (e.g., "success", "error").
            message (str): A message providing additional information about the response.
            event_requested (dict): The event details that were requested by the user.
            calendar_insights (dict, optional): Insights from the calendar regarding the event. Defaults to None.

        Returns:
            dict: The processed response containing the status, message, event details, and any calendar insights.
        """
        response = {
            "status": status,
            "message": message,
            "event_requested": event_requested,
            "calendar_insights": calendar_insights
        }
        return response
    
    @staticmethod
    def convert_to_iso(og_target_dates: list[tuple[str, Optional[str]]]) -> list[tuple[str, Optional[str]]]:
        """Convert original target dates to ISO format.

        Args:
            og_target_dates (list[tuple[str, Optional[str]]]): Original target dates.

        Returns:
            list[tuple[str, Optional[str]]]: Converted target dates in ISO format.
        """
        pdt = pytz.timezone('America/Los_Angeles')
        iso_dates = []
        for start, end in og_target_dates:
            try:
                start_dt = datetime.strptime(start, "%A, %B %d, %Y %I:%M %p")
                start_pdt = pdt.localize(start_dt).isoformat()
                if end:
                    end_dt = datetime.strptime(end, "%A, %B %d, %Y %I:%M %p")
                    end_pdt = pdt.localize(end_dt).isoformat()
                else:
                    end_pdt = None
                logger.info(f"Converted {start} and {end} to PDT: {start_pdt}, {end_pdt}")
            except Exception as e:
                logger.error(f"Datetimes do not conform to format: {e}")
                start_pdt = start
                end_pdt = end

            iso_dates.append((start_pdt, end_pdt))
        return iso_dates
    
    @staticmethod
    async def generate_events(input_request: InputRequest) -> list[dict]:
        """Fetch events from the calendar based on user input.

        Args:
            input_request (InputRequest): The user input request containing the input text.

        Returns:
            list[dict]: A list of dictionaries containing the fetched events.
        """
        input_text = input_request.input_text

        response_handler = HandleResponse(input_text)
        events = response_handler.process_response() 
        logger.info(f"Generated events: {events}")
        return events
    
    @staticmethod
    async def process_input(
        user_id: str,
        events: list[dict]) -> list[Dict[str, Any]]:
        """Processes user input, generates, a response, and returns that response as a list of dictionaries for either further processing or user feedback.

        Args:
            input_request (list[dict]): The user input request containing the input text.

        Raises:
            HTTPException: If an error occurs while processing the input request.

        Returns:
            list[Dict[str, Any]]: A list of dictionaries containing the response for each processed event. 
            This is needed if the system needs to request additional information
        """
        try:
            service = await UniformInterface(user_id).fetch_service()
            await user_config.get_client()
            await project_config.get_client()
            user_data = await user_config.get_single_doc({"user_id": user_id})
            requests = []
            for event in events:
                event_output = event["Event Output"]
                calendar_event = event["Calendar Event"]
                calendar_event = RequestSetup(
                    calendar_event,
                    EventOutput(),
                    user_id,
                    service
                ).tie_project(user_data)
                logger.info(f"Calendar Event: {calendar_event}")
                if event_output.action.lower() == "add":
                    add_handler = AddToCalendar(
                        calendar_event, 
                        event_output, 
                        user_id, 
                        service)
                    add_handler.add_event()
                    response = add_handler.event_output.feature_response

                    responseRequest = MainModel.format_response("completed", response, event)
                    requests.append(responseRequest)

                elif event_output.intent.lower() == "delete":
                    delete_handler = DeleteFromCalendar(
                        calendar_event,
                        event_output,
                        user_id,
                        service
                    )
                    delete_handler.find_matching_events()
                    calendar_insights_dict = {
                        "matching_events": delete_handler.calendar_insights.matching_events,
                    }

                    responseRequest = MainModel.format_response(
                        "request event ID", 
                        f"Please select an event to delete as {delete_handler.calendar_event.event_name}", 
                        event, 
                        calendar_insights_dict)
                    requests.append(responseRequest)

                elif event_output.intent.lower() == "update":
                    update_handler = UpdateFromCalendar(
                        calendar_event,
                        event_output,
                        user_id,
                        service
                    )
                    update_handler.find_matching_events()
                    calendar_insights_dict = {
                        "matching_events": update_handler.calendar_insights.matching_events,
                    }
                    update_handler.calendar_event.datetime_obj.target_datetimes = \
                    [(start, end) for start, end in update_handler.calendar_event.datetime_obj.target_datetimes]
                    responseRequest = MainModel.format_response(
                        "request event ID", 
                        f"Please select an event to update as {update_handler.calendar_event.event_name}", 
                        event, 
                        calendar_insights_dict)
                    requests.append(responseRequest)

                else:
                    result = f"Unknown action '{event_output.intent}' for event '{calendar_event.event_name}'."
                    responseRequest = MainModel.format_response("failed", result, event)
                    requests.append(responseRequest)

            return requests
        except Exception as e:
            logger.error(f"I'm sorry, something went wrong. Please try again: {e}")
            return [{"status": "failed", "message": f"Something went wrong please try again."}]

    @staticmethod
    async def delete_event(event_id: str, request_body: dict) -> dict:
        """Delete an event from the calendar.

        Args:
            event_id (str): The ID of the event to delete.
            request_body (dict): The request body containing event details and calendar insights.

        Raises:
            HTTPException: If an error occurs while deleting the event.

        Returns:
            dict: A dictionary containing the status and message of the delete operation.
        """
        try:
            event_requested = request_body.get("event_requested", {})
            datetime_set = DateTimeSet(
                target_datetimes=event_requested.get("target_dates", [])
            )
            event_output = EventOutput(
                intent=event_requested.get("action", "None"),
                feature_response=event_requested.get("response", "None")
            )
            calendar_event = CalendarEvent(
                event_name=event_requested.get("event_name", None),
                datetime_obj=datetime_set,
                event_id=event_id
            )

            if calendar_event.event_name == "None" or not calendar_event.event_name:
                raise ValueError("Event name must be provided.")
            
            user_id = request_body.get("user_id", "None")
            service = await UniformInterface(user_id).fetch_service()
            calendar_insights_dict = request_body.get("calendar_insights", {})        
            delete_handler = DeleteFromCalendar(
                calendar_event, 
                event_output,
                user_id,
                service)
            delete_handler.calendar_insights = CalendarInsights(**calendar_insights_dict)

            delete_handler.delete_event(event_id)  
            return {"status": "success", "message": event_output.feature_response}
        except Exception as e:
            logger.error(f"I'm sorry, something went wrong. Please try again: {e}")
            return {"status": "failed", "message": f"Something went wrong please try again."}
        
    @staticmethod
    async def update_event(event_id: str, request_body: dict) -> dict:
        """Update an existing event in the calendar.

        Args:
            event_id (str): ID of the event to update.
            request_body (dict): The updated event details.
                {
                    "event_name": "Updated Event",
                    "target_dates": ["2023-10-01T10:00:00Z", "2023-10-01T11:00:00Z"],
                    "action": "update",
                    "response": "Event updated successfully"
                }

        Raises:
            HTTPException: If an error occurs while updating the event.

        Returns:
            dict: A dictionary containing the status and message of the update operation.
        """
        try:
            event_requested = request_body.get("event_requested", {})
            target_dates = MainModel.convert_to_iso(event_requested.get("target_dates", []))
            datetime_set = DateTimeSet(
                target_datetimes=target_dates
            )
            event_output = EventOutput(
                intent=event_requested.get("action", "None"),
                feature_response=event_requested.get("response", "None")
            )
            calendar_event = CalendarEvent(
                event_name=event_requested.get("event_name", None),
                datetime_obj=datetime_set,
                event_id=event_id
            )
            logger.info(f"Event Details Target Datetimes: {calendar_event.datetime_obj.target_datetimes}")

            calendar_insights_dict = request_body.get("calendar_insights", {})
            user_id = request_body.get("user_id", "None")
            service = await UniformInterface(user_id).fetch_service()

            update_handler = UpdateFromCalendar(
                calendar_event,
                event_output,
                user_id,
                service
                )
            update_handler.calendar_insights = CalendarInsights(**calendar_insights_dict)
            update_handler.event_output = event_output
            logger.info(f"Event Details: {update_handler.event_output}")
            logger.info(f"Calendar Insights: {update_handler.calendar_insights}")

            if len(target_dates) > 1:
                update_handler.eliminate_targets(event_id)

            update_handler.update_event(event_id)

            return {"status": "success", "message": event_output.feature_response}
        except Exception as e:
            logger.error(f"I'm sorry, something went wrong. Please try again: {e}")
            return {"status": "failed", "message": f"Something went wrong please try again."}