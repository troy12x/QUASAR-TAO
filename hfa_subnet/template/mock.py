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
        return self.block

    @property
    def block(self):
        # 1 block every 12 seconds
        return 1000 + int((time.time() - self._start_time) / 12)

    def get_subnet_hyperparameters(self, netuid: int, block: int = None):
        # Return a mock object mimicking SubnetHyperparameters
        class MockHyperparams:
            rho = 10
            kappa = 32767
            immunity_period = 0
            min_allowed_weights = 0
            max_weight_limit = 65535
            tempo = 99
            min_difficulty = 0
            max_difficulty = 1000000
            weights_version = 0
            weights_rate_limit = 0
            adjustment_interval = 0
            activity_cutoff = 0
            registration_allowed = True
            target_regs_per_interval = 0
            min_burn = 0
            max_burn = 0
            bonds_moving_avg = 0
            max_regs_per_block = 0
            serving_rate_limit = 0
            max_validators = 1
            difficulty = 0
            adjustment_alpha = 0
            
        return MockHyperparams()

    def subnet_exists(self, netuid: int) -> bool:
        return True

    def neurons_lite(self, netuid: int, block: int = None):
        print(f"DEBUG: MockSubtensor.neurons_lite called with netuid={netuid}")
        return [self.neuron_for_uid_lite(uid, netuid, block) for uid in range(self.n)]

    def force_register_neuron(self, netuid, hotkey_ss58, coldkey_ss58, balance, stake):
        # No-op in this fully mocked version as we generate neurons on the fly in neurons_lite
        print(f"DEBUG: Mock register neuron for {hotkey_ss58} (Mocked success)")
        return True

    def neuron_for_uid_lite(self, uid, netuid, block=None):
        netuid = int(netuid)
        
        # Determine hotkey
        if uid == 0:
            hotkey = self.validator_hotkey
        else:
            hotkey = f"miner-hotkey-{uid}"
            
        return bt.NeuronInfoLite(
            netuid=netuid,
            uid=uid,
            hotkey=hotkey,
            coldkey="mock-coldkey",
            axon_info=bt.AxonInfo(
                version=1,
                ip="127.0.0.1",
                port=8091,
                ip_type=4,
                hotkey=hotkey,
                coldkey="mock-coldkey",
            ),
            prometheus_info=None,
            stake=100000,
            rank=0,
            emission=0,
            active=1,
            stake_dict={},
            total_stake=100000,
            incentive=0,
            consensus=0,
            trust=0,
            dividends=0,
            last_update=0,
            validator_permit=True,
            validator_trust=0,
            pruning_score=0,
        )

    def serve_axon(self, netuid, axon, wait_for_inclusion=False, wait_for_finalization=False, certificate=None):
        print(f"DEBUG: Mock serve_axon called for netuid {netuid}")
        return True

    def set_weights(self, netuid, wallet, uids, weights, version_key=0, wait_for_inclusion=False, wait_for_finalization=False):
        print(f"DEBUG: Mock set_weights called for netuid {netuid} with {len(uids)} weights")
        return True

    def get_balance(self, address):
        return bt.Balance(1000)

    def __init__(self, netuid, n=16, wallet=None, network="mock"):
        super().__init__(network=network)
        self._start_time = time.time()
        self.netuid = int(netuid)
        self.n = n

        if wallet is not None:
            self.validator_hotkey = wallet.hotkey.ss58_address
        else:
            self.validator_hotkey = "validator-hotkey"

        print(f"DEBUG: MockSubtensor init. Netuid: {self.netuid}. Neurons: {self.n}")


class MockMetagraph(bt.Metagraph):
    @property
    def hotkeys(self):
        return self._hotkeys

    @hotkeys.setter
    def hotkeys(self, value):
        self._hotkeys = value

    @property
    def uids(self):
        return self._uids

    @uids.setter
    def uids(self, value):
        self._uids = value

    @property
    def axons(self):
        return self._axons

    @axons.setter
    def axons(self, value):
        self._axons = value

    @property
    def last_update(self): return self._last_update
    @last_update.setter
    def last_update(self, value): self._last_update = value

    @property
    def S(self): return self._S
    @S.setter
    def S(self, value): self._S = value

    @property
    def R(self): return self._R
    @R.setter
    def R(self, value): self._R = value

    @property
    def I(self): return self._I
    @I.setter
    def I(self, value): self._I = value

    @property
    def E(self): return self._E
    @E.setter
    def E(self, value): self._E = value

    @property
    def C(self): return self._C
    @C.setter
    def C(self, value): self._C = value

    @property
    def T(self): return self._T
    @T.setter
    def T(self, value): self._T = value

    @property
    def W(self): return self._W
    @W.setter
    def W(self, value): self._W = value

    @property
    def dividends(self): return self._dividends
    @dividends.setter
    def dividends(self, value): self._dividends = value

    @property
    def incentives(self): return self._incentives
    @incentives.setter
    def incentives(self, value): self._incentives = value

    @property
    def consensus(self): return self._consensus
    @consensus.setter
    def consensus(self, value): self._consensus = value

    @property
    def trust(self): return self._trust
    @trust.setter
    def trust(self, value): self._trust = value

    @property
    def rank(self): return self._rank
    @rank.setter
    def rank(self, value): self._rank = value

    def __init__(self, netuid=1, network="mock", subtensor=None):
        netuid = int(netuid)
        super().__init__(netuid=netuid, network=network, sync=False)

        if subtensor is not None:
            self.subtensor = subtensor
        self.sync(subtensor=subtensor)

        for axon in self.axons:
            axon.ip = "127.0.0.0"
            axon.port = 8091

        bt.logging.info(f"Metagraph: {self}")
        bt.logging.info(f"Axons: {self.axons}")
        
    def sync(self, block=None, lite=True, subtensor=None):
        if subtensor is None:
            subtensor = self.subtensor
            
        if subtensor is None:
            return

        # Explicitly use integer netuid
        netuid = int(self.netuid)
        
        # Helper to get attributes safe
        self.neurons = subtensor.neurons_lite(netuid=netuid, block=block)
        self.n = len(self.neurons)
        self._hotkeys = [n.hotkey for n in self.neurons]
        self._uids = [n.uid for n in self.neurons]
        self._axons = [n.axon_info for n in self.neurons]
        
        # Use current subtensor block to satisfy sync conditions
        # distinct from mock block 0 which causes loop
        current_block = subtensor.block
        self.block = current_block
        self._last_update = [current_block] * self.n
        
        # Set other attributes to empty/zeros to prevent attribute errors
        self._S = [0.0] * self.n
        self._R = [0.0] * self.n
        self._I = [0.0] * self.n
        self._E = [0.0] * self.n
        self._C = [0.0] * self.n
        self._T = [0.0] * self.n
        self._W = [[0.0] * self.n] * self.n
        self._dividends = [0.0] * self.n
        self._incentives = [0.0] * self.n
        self._consensus = [0.0] * self.n
        self._trust = [0.0] * self.n
        self._rank = [0.0] * self.n

    def __str__(self):
        return f"MockMetagraph(netuid:{self.netuid}, n:{self.n}, block:{self.block}, network:{self.network})"

    def __repr__(self):
        return self.__str__()


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
