from transformers import GPT2LMHeadModel, GPT2Tokenizer
weights_dir = '/dist/model_output/checkpoint-1699'

def get_model_tokenizer(weights_dir, device = 'cpu'):
    weights_dir = '/home/joe/dist/model_output/checkpoint-1699'
    print("Loading Model ...")
    model = GPT2LMHeadModel.from_pretrained(weights_dir)
    model.to('cpu')
    print("Model Loaded ...")
    tokenizer = GPT2Tokenizer.from_pretrained(weights_dir)
    return model, tokenizer

def generate_messages(
    model,
    tokenizer,
    prompt_text,
    stop_token,
    length,
    num_return_sequences,
    temperature = 0.8,
    k=20,
    p=0.9,
    repetition_penalty = 1.0,
    device = 'cpu'
):

    MAX_LENGTH = int(10000)
    def adjust_length_to_model(length, max_sequence_length):

        if length < 0 and max_sequence_length > 0:
            length = max_sequence_length
        elif 0 < max_sequence_length < length:
            length = max_sequence_length  # No generation bigger than model size
        elif length < 0:
            length = MAX_LENGTH  # avoid infinite loop
        return length

    length = adjust_length_to_model(length=length, max_sequence_length=model.config.max_position_embeddings)

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

    encoded_prompt = encoded_prompt.to(device)

    output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=length + len(encoded_prompt[0]),
            temperature=temperature,
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=num_return_sequences,
        )

    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []
    global dist_out
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        #print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(stop_token) if stop_token else None]
        print(text)

        dist_out = text
        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )
        #prompt_text +

        generated_sequences.append(total_sequence)
    return generated_sequences

model, tokenizer = get_model_tokenizer(weights_dir, device = 'cpu')

temperature = .7
k=400
p=0.9
repetition_penalty = 1.1
num_return_sequences = 1
length = 128
stop_token = '***'
#prompt_text = "USER: Cinder, "+input()+".<END>"+"\nCINDER: "

previous =""
num_chat = 1
while num_chat <=20:


    ques = previous + 'USER: ' + input('USER: ') + '<END>\nCINDER: '
    num_chat += 1
    inp = ques
    prompt_text = inp

    gout = generate_messages(
        model,
        tokenizer,
        prompt_text,
        stop_token,
        length,
        num_return_sequences,
        temperature = temperature,
        k=k,
        p=p,
        repetition_penalty = repetition_penalty,

    )

    string = str(gout)

    list = string.split('<END>')
    for item in list:
        item.strip()
    first_out = str(list[0])
    first_out = str(first_out)
    print('first_out: ' + first_out)
    #while item in list == item.startswith('CINDER:'):

    qa_out = ques + first_out + '<END>'



    cinder = str(qa_out)
    print('cinder: ' + cinder)
    out = ''
    if len(cinder) > 510:
      for i in range(len(cinder)):
          if i > 256:
              out = out + cinder[i]
    else:
      out = cinder
