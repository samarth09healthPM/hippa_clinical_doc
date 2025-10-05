from cryptography.fernet import Fernet
key = Fernet.generate_key()
with open("secure_store/fernet.key", "wb") as f:
    f.write(key)
print("Key written to secure_store/fernet.key")
