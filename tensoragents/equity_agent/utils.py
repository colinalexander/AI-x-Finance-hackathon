import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple
from uuid import UUID
import neatplot
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorzero import AsyncTensorZeroGateway
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)
neatplot.set_style("notex")

class TensorZeroEquityAgent:
    
    def __init__(self, client: AsyncTensorZeroGateway, variant_name: Optional[str] = None):
        self.client = client
        self.variant_name = variant_name
    
    async def trade_evaluator(self, data, episode_id: Optional[UUID] = None) -> Tuple[str, Optional[UUID]]:
        trade_decisions = ['Long', 'Neutral', 'Short']

        try: 
            result = await self.client.inference(
                function_name="financial_data_evaluator",
                input = {}, 
                variant_name = self.variant_name, 
                episode_id = episode_id
            )

            thinking = result.output.parsed["thinking"]
            logger.info(f"Trade evaluator thinking: {thinking}")
            move = result.output.parsed["move"]
            logger.info(f"Trade evaluator: {move}")
            episode_id = result.episode_id
        except Exception as e:
            logger.error(f"Error occurred: {type(e).__name__}: {e}")
            logger.info("Choosing a random move as fallback.")
            move = random.choice(trade_decisions)
            return move, episode_id
        return move, episode_id