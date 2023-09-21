import mailbox
import pandas as pd
from email.utils import parseaddr
from nltk.tokenize import sent_tokenize
from .utils import ATTACH_N_WORD_THRESHOLD, chunk_text, ensure_valid_encoding

def get_elements(mbox_file):
    def extract_payload(msg):
        if msg.is_multipart():
            return ''.join(part.get_payload(decode=True).decode('utf-8', 'ignore') for part in msg.get_payload() if part.get_content_type() == 'text/plain')
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                return payload.decode('utf-8', 'ignore')
            else:
                return ""

    mbox = mailbox.mbox(mbox_file)

    temp = []
    for msg in mbox:
        subject = msg['Subject']
        date = msg['Date']
        payload = extract_payload(msg)
        from_name, from_email = parseaddr(msg['From'])
        to_name, to_email = parseaddr(msg['To'])
        cc = msg.get('Cc')
        bcc = msg.get('Bcc')

        temp.append((subject, date, payload, from_name, from_email, to_name, to_email, cc, bcc))

    return temp, True

def create_train_df(elements):
    df = pd.DataFrame(index=range(len(elements)), columns=["para", "filename", "display"])

    for i, elem in enumerate(elements):
        subject = elem[0]
        from_name = elem[3]
        payload = elem[2]

        sents = sent_tokenize(payload)
        sents = [sent.replace("\t", " ").replace(",", " ").replace("\n", " ").strip().lower() for sent in sents]
        para = " ".join(sents)

        # Format "display" column
        display_text = f"{subject}\n{from_name}\n{payload[:100]}..."
        display = ensure_valid_encoding(display_text)

        df.iloc[i] = [ensure_valid_encoding(para), subject, display]

    return df
# def create_train_df(elements):
#     # Define columns
#     columns = ["Subject", "Date", "Payload", "From_Name", "From_Email", "To_Name", "To_Email", "Cc", "Bcc", "Processed_Payload", "Display"]
#     df = pd.DataFrame(elements, columns=columns[:-2])  # Initialize dataframe with original columns

#     # Process payload to create 'Processed_Payload' and 'Display' columns
#     for i, elem in enumerate(elements):
#         payload = elem[2]  # The "Payload" field
#         sents = sent_tokenize(payload)
#         sents = [sent.replace("\t", " ").replace(",", " ").replace("\n", " ").strip().lower() for sent in sents]
#         processed_payload = " ".join(sents)
#         display = payload.replace("\n", " ")

#         df.at[i, "Processed_Payload"] = ensure_valid_encoding(processed_payload)
#         df.at[i, "Display"] = ensure_valid_encoding(display)

#     return df