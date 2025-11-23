# ======================================================
# COFFEE SHOP VOICE AGENT  
# Developer: Prabhu Krupa Dwibedy
# ======================================================

import logging
import json
import os
import asyncio
from datetime import datetime
from typing import Annotated, Literal
from dataclasses import dataclass, field

print("\n" + "=" * 50)
print("COFFEE SHOP AGENT - STARBUCKS")
print("agent.py LOADED SUCCESSFULLY")
print("=" * 50 + "\n")

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    tokenize,
    metrics,
    MetricsCollectedEvent,
    RunContext,
    function_tool,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# ORDER MANAGEMENT SYSTEM
# ======================================================
@dataclass
class OrderState:
    drinkType: str | None = None
    size: str | None = None
    milk: str | None = None
    extras: list[str] = field(default_factory=list)
    name: str | None = None
    
    def is_complete(self) -> bool:
        return all([
            self.drinkType is not None,
            self.size is not None,
            self.milk is not None,
            self.extras is not None,
            self.name is not None
        ])
    
    def to_dict(self) -> dict:
        return {
            "drinkType": self.drinkType,
            "size": self.size,
            "milk": self.milk,
            "extras": self.extras,
            "name": self.name
        }
    
    def get_summary(self) -> str:
        if not self.is_complete():
            return "Order in progress..."
        
        extras_text = f" with {', '.join(self.extras)}" if self.extras else ""
        return f"{self.size.upper()} {self.drinkType.title()} with {self.milk.title()} milk{extras_text} for {self.name}"

@dataclass
class Userdata:
    order: OrderState
    session_start: datetime = field(default_factory=datetime.now)

# ======================================================
# BARISTA AGENT FUNCTION TOOLS
# ======================================================

@function_tool
async def set_drink_type(
    ctx: RunContext[Userdata],
    drink: Annotated[
        Literal["latte", "cappuccino", "americano", "espresso", "mocha", "coffee", "cold brew", "matcha"],
        Field(description="The type of coffee drink the customer wants"),
    ],
) -> str:
    ctx.userdata.order.drinkType = drink
    print(f"DRINK SET: {drink.upper()}")
    print(f"Order Progress: {ctx.userdata.order.get_summary()}")
    return f"One {drink} coming up!"

@function_tool
async def set_size(
    ctx: RunContext[Userdata],
    size: Annotated[
        Literal["small", "medium", "large", "extra large"],
        Field(description="The size of the drink"),
    ],
) -> str:
    ctx.userdata.order.size = size
    print(f"SIZE SET: {size.upper()}")
    print(f"Order Progress: {ctx.userdata.order.get_summary()}")
    return f"{size.title()} size selected."

@function_tool
async def set_milk(
    ctx: RunContext[Userdata],
    milk: Annotated[
        Literal["whole", "skim", "almond", "oat", "soy", "coconut", "none"],
        Field(description="The type of milk for the drink"),
    ],
) -> str:
    ctx.userdata.order.milk = milk
    print(f"MILK SET: {milk.upper()}")
    print(f"Order Progress: {ctx.userdata.order.get_summary()}")
    
    if milk == "none":
        return "Black coffee confirmed."
    return f"{milk.title()} milk selected."

@function_tool
async def set_extras(
    ctx: RunContext[Userdata],
    extras: Annotated[
        list[Literal["sugar", "whipped cream", "caramel", "extra shot", "vanilla", "cinnamon", "Honey"]] | None,
        Field(description="List of extras, or empty for no extras"),
    ] = None,
) -> str:
    ctx.userdata.order.extras = extras if extras else []
    print(f"EXTRAS SET: {ctx.userdata.order.extras}")
    print(f"Order Progress: {ctx.userdata.order.get_summary()}")
    
    if ctx.userdata.order.extras:
        return f"Added {', '.join(ctx.userdata.order.extras)}."
    return "No extras added."

@function_tool
async def set_name(
    ctx: RunContext[Userdata],
    name: Annotated[str, Field(description="Customer's name for the order")],
) -> str:
    ctx.userdata.order.name = name.strip().title()
    print(f"NAME SET: {ctx.userdata.order.name}")
    print(f"Order Progress: {ctx.userdata.order.get_summary()}")
    return f"Thank you, {ctx.userdata.order.name}."

@function_tool
async def complete_order(ctx: RunContext[Userdata]) -> str:
    order = ctx.userdata.order
    
    if not order.is_complete():
        missing = []
        if not order.drinkType: missing.append("drink type")
        if not order.size: missing.append("size")
        if not order.milk: missing.append("milk")
        if order.extras is None: missing.append("extras")
        if not order.name: missing.append("name")
        
        print(f"CANNOT COMPLETE - Missing: {', '.join(missing)}")
        return f"Missing: {', '.join(missing)}"
    
    print(f"ORDER READY: {order.get_summary()}")
    
    try:
        save_order_to_json(order)
        return f"Your {order.size} {order.drinkType} with {order.milk} milk is confirmed, {order.name}."
        
    except Exception as e:
        print(f"ORDER SAVE FAILED: {e}")
        return "Order recorded but encountered an issue."

@function_tool
async def get_order_status(ctx: RunContext[Userdata]) -> str:
    order = ctx.userdata.order
    if order.is_complete():
        return f"Order complete: {order.get_summary()}"
    
    return f"Order in progress: {order.get_summary()}"

class BaristaAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are a friendly and professional barista at Starbucks.
            
            Collect the customer's:
            - Drink type
            - Size
            - Milk choice
            - Extras
            - Name
            
            Ask one question at a time and use function tools to store data.
            """,
            tools=[
                set_drink_type,
                set_size,
                set_milk,
                set_extras,
                set_name,
                complete_order,
                get_order_status,
            ],
        )

def create_empty_order():
    return OrderState()

# ======================================================
# ORDER STORAGE
# ======================================================
def get_orders_folder():
    base_dir = os.path.dirname(__file__)
    backend_dir = os.path.abspath(os.path.join(base_dir, ".."))
    folder = os.path.join(backend_dir, "orders")
    os.makedirs(folder, exist_ok=True)
    return folder

def save_order_to_json(order: OrderState) -> str:
    print("Saving order...")
    folder = get_orders_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"order_{timestamp}.json"
    path = os.path.join(folder, filename)

    try:
        order_data = order.to_dict()
        order_data["timestamp"] = datetime.now().isoformat()
        order_data["session_id"] = f"session_{timestamp}"
        
        with open(path, "w", encoding='utf-8') as f:
            json.dump(order_data, f, indent=4, ensure_ascii=False)
        
        print("ORDER SAVED:", path)
        return path
        
    except Exception as e:
        print("ERROR:", e)
        raise e

# ======================================================
# SYSTEM VALIDATION
# ======================================================
def test_order_saving():
    print("Testing order saving...")
    
    test_order = OrderState()
    test_order.drinkType = "latte"
    test_order.size = "medium"
    test_order.milk = "oat"
    test_order.extras = ["extra shot", "vanilla"]
    test_order.name = "TestCustomer"
    
    try:
        path = save_order_to_json(test_order)
        print("SUCCESS:", path)
        return True
    except Exception as e:
        print("FAILED:", e)
        return False

# ======================================================
# INITIALIZATION
# ======================================================
def prewarm(proc: JobProcess):
    print("Prewarming VAD model...")
    proc.userdata["vad"] = silero.VAD.load()
    print("VAD model loaded.")

# ======================================================
# MAIN ENTRYPOINT
# ======================================================
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    print("Starbucks Agent Ready")
    print("Orders folder:", get_orders_folder())

    test_order_saving()

    userdata = Userdata(order=create_empty_order())
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("New session:", session_id)

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    usage_collector = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)

    await session.start(
        agent=BaristaAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

# ======================================================
# APPLICATION BOOTSTRAP
# ======================================================
if __name__ == "__main__":
    print("Starting Starbucks Coffee Agent...")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
