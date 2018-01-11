from SerializableStateObservation import SerializableStateObservation


class State:
    def __init__(self, sso: 'SerializableStateObservation'):
        self.sso = sso

    def get_state_from_sso(self):
        pass