from google import genai
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
import logging
import hashlib
from tqdm import tqdm
import time
from filelock import FileLock


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _secret_fingerprint(secret):
    """
    Return a short stable fingerprint for debugging without exposing the real key.
    This does NOT reveal any character of the API key.
    """
    if not secret:
        return "<empty>"
    return hashlib.sha256(str(secret).encode("utf-8")).hexdigest()[:8]


def _redact_secret(message, secret):
    """
    Remove the API key from an error message if the SDK accidentally includes it.
    """
    if message is None:
        return ""

    message = str(message)

    if secret:
        message = message.replace(str(secret), "<REDACTED_API_KEY>")

    return message


class LLMEmbAPI:
    def __init__(self, datadir, LLMEmb_config):
        self.LLMEmb_config = LLMEmb_config
        cwd = os.getcwd()
        start = time.time()
        self.LLMEmb_user_prompts = os.path.join(cwd, 'data', datadir, 'LLMEmb_output', 'user_prompts.pkl')
        self.LLMEmb_item_prompts = os.path.join(cwd, 'data', datadir, 'LLMEmb_output', 'item_prompts.pkl')
        self.api_keys = self.LLMEmb_config.api_keys
        print("Generating Emb API key cycle..\t")
        self.api_key_cycle = cycle(self.api_keys)  # Cycle through API keys if needed
        self.save_calls_steps = self.LLMEmb_config.save_calls_steps
        self.max_workers = self.LLMEmb_config.threads
        print(f"Max workers has been set to: {self.max_workers} ")
        self.user_responses_file = os.path.join(cwd, 'data', datadir, 'LLMEmb_output', 'api_user_responses.pkl')
        self.item_responses_file = os.path.join(cwd, 'data', datadir, 'LLMEmb_output', 'api_item_responses.pkl')
        self.user_responses = {}
        self.item_responses = {}

        if self.LLMEmb_config.resume:
            self.user_responses = self._load_existing_user_responses()
        if self.LLMEmb_config.resume:
            self.item_responses = self._load_existing_item_responses()
        # cycle with different API clients
        api_client_cycle = []
        print("Generating Emb API-Client cycle..")
        for api in self.api_keys:
            client = genai.Client(api_key=api)
            api_client_cycle.append(client)
        self.api_client_cycle = cycle(api_client_cycle)
        self.user_prompts = self._load_user_prompts()
        self.item_prompts = self._load_item_prompts()
        print(f"Emb setup completed in {time.time() - start:.2f}s\n")

    def _load_user_prompts(self):
        print(f"Loading Emb prompts..")
        if os.path.exists(self.LLMEmb_user_prompts):
            try:
                with open(self.LLMEmb_user_prompts, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading Emb pre-prompts from: {self.LLMEmb_user_prompts}\n"
                             f"ERROR CODE: {str(e)}")
        return {}

    def _load_item_prompts(self):
        print(f"Loading Emb prompts..")
        if os.path.exists(self.LLMEmb_item_prompts):
            try:
                with open(self.LLMEmb_item_prompts, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading Emb pre-prompts from: {self.LLMEmb_item_prompts}\n"
                             f"ERROR CODE: {str(e)}")
        return {}

    def _load_existing_user_responses(self):
        print(f"Loading existing responses from {self.user_responses_file}..")
        if os.path.exists(self.user_responses_file):
            try:
                with open(self.user_responses_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading existing responses: {str(e)}")
        return {}

    def _load_existing_item_responses(self):
        print(f"Loading existing responses from {self.item_responses_file}..")
        if os.path.exists(self.item_responses_file):
            try:
                with open(self.item_responses_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading existing responses: {str(e)}")
        return {}

    def _atomic_save_user(self, data):
        temp_path = self.user_responses_file + ".tmp"
        lock = FileLock(temp_path + ".lock")

        with lock:
            # remove any temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            with open(temp_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # rename the temporary file to the original file
            os.replace(temp_path, self.user_responses_file)
            logger.info(f"Saved {len(data)} responses to {self.user_responses_file}")

    def _atomic_save_item(self, data):
        temp_path = self.item_responses_file + ".tmp"
        lock = FileLock(temp_path + ".lock")

        with lock:
            # remove any temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            with open(temp_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # rename the temporary file to the original file
            os.replace(temp_path, self.item_responses_file)
            logger.info(f"Saved {len(data)} responses to {self.item_responses_file}")

    def _call_gemini_api(self, prompt, uid):
        # Try using available API keys until a call succeeds.
        while True:
            api_key = None

            try:
                api_key = next(self.api_key_cycle)
                client = next(self.api_client_cycle)

                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt
                )

                key_fp = _secret_fingerprint(api_key)
                logger.info(
                    "Response for: %s... using Gemini API key fingerprint (HASH): %s",
                    uid,
                    key_fp
                )

                return response

            except Exception as e:
                raw_error = str(e)
                safe_error = _redact_secret(raw_error, api_key)
                key_fp = _secret_fingerprint(api_key)

                logger.error(
                    "API call failed with Gemini API key fingerprint (HASH) %s for %s: %s",
                    key_fp,
                    uid,
                    safe_error
                )

                if '503' in raw_error or '429' in raw_error:
                    logger.info("Retrying API call for %s due to error", uid)
                    time.sleep(5)
                    continue

                break

        raise Exception(f"All retries failed for user {uid}.")


    def process_user_prompts(self):
        # skip the prompts that have already been processed
        self.user_prompts = self._load_user_prompts()
        tasks = {uid: prompt for uid, prompt in self.user_prompts.items() if uid not in self.user_responses}
        total_tasks = len(self.user_prompts)
        logger.info(f"Processing {len(tasks)} new prompts out of {total_tasks} total.")

        count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_uid = {executor.submit(self._call_gemini_api, prompt, uid): uid for uid, prompt in tasks.items()}

            for future in tqdm(as_completed(future_to_uid), total=len(future_to_uid), desc="Processing prompts"):
                uid = future_to_uid[future]
                try:
                    res = future.result()
                    self.user_responses[uid] = res
                    count += 1
                except Exception as e:
                    logger.error(f"Failed processing {uid}: {str(e)}")

                # incremental save after every checkpoint
                if count % self.save_calls_steps == 0:
                    self._atomic_save_user(self.user_responses)

        # final save after processing all prompts
        self._atomic_save_user(self.user_responses)
        return self.user_responses

    def process_single_user_prompt(self, user_id):
        # if the user_id exists
        if user_id not in self.user_prompts:
            logger.warning(f"User ID {user_id} not found in prompts.")
            return self.user_responses

        prompt = self.user_prompts[user_id]
        logger.info(f"Fixing prompt for user {user_id}")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._call_gemini_api, prompt, user_id)
            try:
                res = future.result()
                self.user_responses[user_id] = res
                self._atomic_save_user(self.user_responses)
            except Exception as e:
                logger.error(f"Failed processing {user_id}: {str(e)}")

        return self.user_responses

    def process_item_prompts(self):
        self.item_prompts = self._load_item_prompts()
        tasks = {iid: prompt for iid, prompt in self.item_prompts.items()
                 if iid not in self.item_responses}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._call_gemini_api, prompt, iid): iid
                for iid, prompt in tasks.items()
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing item prompts"):
                iid = futures[future]
                try:
                    self.item_responses[iid] = future.result()
                except Exception as e:
                    logger.error(f"Item {iid} failed: {e}")

                if len(self.item_responses) % self.save_calls_steps == 0:
                    self._atomic_save_item(self.item_responses)

        self._atomic_save_item(self.item_responses)

    def process_single_item_prompt(self, item_id):
        # if the item_id exists
        if item_id not in self.item_prompts:
            logger.warning(f"Item ID {item_id} not found in prompts.")
            return self.item_responses

        prompt = self.item_prompts[item_id]
        logger.info(f"Fixing prompt for item {item_id}")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._call_gemini_api, prompt, item_id)
            try:
                res = future.result()
                self.item_responses[item_id] = res
                self._atomic_save_item(self.item_responses)
            except Exception as e:
                logger.error(f"Failed processing {item_id}: {str(e)}")

        return self.item_responses