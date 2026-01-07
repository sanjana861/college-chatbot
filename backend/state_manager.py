SESSION_STATE = {}

def get_state(sid):
    return SESSION_STATE.get(sid)

def set_state(sid, state):
    SESSION_STATE[sid] = state

def clear_state(sid):
    SESSION_STATE.pop(sid, None)
