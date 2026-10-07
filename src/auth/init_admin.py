import getpass
import os
import sys

from src.auth.database import SessionLocal, init_db
from src.auth.models import Role
from src.users import crud


def main():
    init_db()
    db = SessionLocal()
    try:
        username = os.getenv("INIT_ADMIN_USERNAME") or input("Nom d'utilisateur admin : ").strip()
        if not username:
            print("Nom d'utilisateur vide — abandon.", file=sys.stderr)
            sys.exit(1)

        if crud.get_user_by_username(db, username):
            print(f"L'utilisateur « {username} » existe déjà — rien à faire.")
            return

        password = os.getenv("INIT_ADMIN_PASSWORD") or getpass.getpass("Mot de passe admin : ")
        if len(password) < 8 or len(password.encode('utf-8')) > 72:
            print("Le mot de passe doit contenir au moins 8 caractères.", file=sys.stderr)
            sys.exit(1)

        crud.create_user(db, username, password, role=Role.ADMIN)
        print(f"✓ Administrateur « {username} » créé avec succès.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
