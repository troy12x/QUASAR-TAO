import time

import asyncio
import random
import bittensor as bt

from typing import List


class MockWallet:
    def __init__(self, config=None, **kwargs):
        self._config = config
        self.name = "mock_wallet"
        self.hotkey_str = "mock_hotkey"
        self._hotkey = None
        self._coldkey = None
        self._coldkeypub = None

    @property
    def hotkey(self):
        if self._hotkey is None:
            self._hotkey = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
        return self._hotkey

    @property
    def coldkey(self):
        if self._coldkey is None:
            self._coldkey = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
        return self._coldkey

    @property
    def coldkeypub(self):
        if self._coldkeypub is None:
            self._coldkeypub = self.coldkey
        return self._coldkeypub
        
    def unlock_hotkey(self):
        return self.hotkey

    def __str__(self):
        return f"MockWallet({self.name}, {self.hotkey.ss58_address})"
        
    def __repr__(self):
        return self.__str__()


class MockSubtensor(bt.MockSubtensor):
    def get_current_block(self):
        return 1000

    @property
    def block(self):
        return 100

    def __init__(self, netuid, n=16, wallet=None, network="mock"):
        super().__init__(network=network)

        print(f"DEBUG: MockSubtensor init. Netuid: {netuid}")
        
        # Force creation of subnet to ensure state is correct
        try:
            print(f"DEBUG: Attempting to create subnet {netuid}...")
            self.create_subnet(netuid)
            print(f"DEBUG: Subnet {netuid} created.")
        except Exception as e:
            print(f"DEBUG: Subnet creation failed (might already exist): {e}")

        # Register ourself (the validator) as a neuron at uid=0
        if wallet is not None:
            print(f"DEBUG: Registering validator {wallet.hotkey.ss58_address}...")
            self.force_register_neuron(
                netuid=netuid,
                hotkey_ss58=wallet.hotkey.ss58_address,
                coldkey_ss58=wallet.coldkey.ss58_address,
                balance=100000,
                stake=100000,
            )

        # Register n mock neurons who will be miners
        print(f"DEBUG: Registering {n} mock miners...")
        for i in range(1, n + 1):
            self.force_register_neuron(
                netuid=netuid,
                hotkey_ss58=f"miner-hotkey-{i}",
                coldkey_ss58="mock-coldkey",
                balance=100000,
                stake=100000,
            )


class MockMetagraph(bt.Metagraph):
    def __init__(self, netuid=1, network="mock", subtensor=None):
        super().__init__(netuid=netuid, network=network, sync=False)

        if subtensor is not None:
            self.subtensor = subtensor
        self.sync(subtensor=subtensor)

        for axon in self.axons:
            axon.ip = "127.0.0.0"
            axon.port = 8091

        bt.logging.info(f"Metagraph: {self}")
        bt.logging.info(f"Axons: {self.axons}")
        
        # Ensure these are real integers, not Mocks
        self.n = len(self.axons)
        self.last_update = [0] * self.n
        self.block = 0


class MockDendrite(bt.Dendrite):
    """
    Replaces a real bittensor network request with a mock request that just returns some static response for all axons that are passed and adds some random delay.
    """

    def __init__(self, wallet):
        super().__init__(wallet)

    async def forward(
        self,
        axons: List[bt.Axon],
        synapse: bt.Synapse = bt.Synapse(),
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
    ):
        if streaming:
            raise NotImplementedError("Streaming not implemented yet.")

        async def query_all_axons(streaming: bool):
            """Queries all axons for responses."""

            async def single_axon_response(i, axon):
                """Queries a single axon for a response."""

                start_time = time.time()
                s = synapse.copy()
                # Attach some more required data so it looks real
                s = self.preprocess_synapse_for_request(axon, s, timeout)
                # We just want to mock the response, so we'll just fill in some data
                process_time = random.random()
                if process_time < timeout:
                    s.dendrite.process_time = str(time.time() - start_time)
                    # Update the status code and status message of the dendrite to match the axon
                    # TODO (developer): replace with your own expected synapse data
                    s.dummy_output = s.dummy_input * 2
                    s.dendrite.status_code = 200
                    s.dendrite.status_message = "OK"
                    synapse.dendrite.process_time = str(process_time)
                else:
                    s.dummy_output = 0
                    s.dendrite.status_code = 408
                    s.dendrite.status_message = "Timeout"
                    synapse.dendrite.process_time = str(timeout)

                # Return the updated synapse object after deserializing if requested
                if deserialize:
                    return s.deserialize()
                else:
                    return s

            return await asyncio.gather(
                *(
                    single_axon_response(i, target_axon)
                    for i, target_axon in enumerate(axons)
                )
            )

        return await query_all_axons(streaming)

    def __str__(self) -> str:
        """
        Returns a string representation of the Dendrite object.

        Returns:
            str: The string representation of the Dendrite object in the format "dendrite(<user_wallet_address>)".
        """
        return "MockDendrite({})".format(self.keypair.ss58_address)
