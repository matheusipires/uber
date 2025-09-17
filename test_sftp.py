import os, sys, getpass
import paramiko

HOST="sftp.uber.com"
PORT=2222
USER="81f6bec5"
KEY_PATH=r"C:\Users\matheus.pires\.ssh\id_rsa_uber_fix"
PASSPHRASE = os.getenv("SFTP_PASSPHRASE")  # defina se sua chave tiver senha

paramiko.util.log_to_file("paramiko.log")  # gera um log detalhado

def load_key(path, passphrase=None):
    loaders = [
        ("RSA", paramiko.RSAKey.from_private_key_file),
        ("ECDSA", paramiko.ECDSAKey.from_private_key_file),
    ]
    # Ed25519 pode não existir em algumas builds antigas
    if hasattr(paramiko, "Ed25519Key"):
        loaders.append(("Ed25519", paramiko.Ed25519Key.from_private_key_file))

    last_err = None
    for name, loader in loaders:
        try:
            print(f"Tentando chave {name}…")
            return loader(path, password=passphrase)
        except Exception as e:
            last_err = e
    raise last_err

def main():
    if not os.path.exists(KEY_PATH):
        print("Arquivo de chave não encontrado:", KEY_PATH)
        sys.exit(1)
    try:
        pkey = load_key(KEY_PATH, PASSPHRASE)
        print("Chave carregada com sucesso.")
    except Exception as e:
        print("Falha ao carregar chave privada:", e)
        sys.exit(1)

    t = paramiko.Transport((HOST, PORT))
    try:
        t.start_client(timeout=15)
        print("Conectado, autenticando…")
        t.auth_publickey(USER, pkey)
        if not t.is_authenticated():
            print("❌ Falha na autenticação (servidor rejeitou a chave).")
            sys.exit(1)
        s = paramiko.SFTPClient.from_transport(t)
        print("✅ Autenticado! Listando diretório raiz:")
        print(s.listdir("/"))
        s.close()
    except paramiko.ssh_exception.BadAuthenticationType as e:
        print("Servidor não aceita esse método de auth:", e.allowed_types)
    except paramiko.ssh_exception.AuthenticationException as e:
        print("❌ Authentication failed:", e)
        print("Causas comuns: chave pública não corresponde, IP fora da allowlist, ou ativação ainda pendente.")
    finally:
        t.close()

if __name__ == "__main__":
    main()
