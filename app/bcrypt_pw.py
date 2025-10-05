import bcrypt
print(bcrypt.hashpw(b"mypassword", bcrypt.gensalt()).decode())