ROLE_LEVEL = {"student": 1, "staff": 2, "admin": 3}


def has_role(user_role: str, required_role: str) -> bool:
    return ROLE_LEVEL.get(user_role, 0) >= ROLE_LEVEL.get(required_role, 999)
