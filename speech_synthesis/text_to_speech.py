"""Synthesizes speech from the input string of text or ssml.

Note: ssml must be well-formed according to:
    https://www.w3.org/TR/speech-synthesis/
"""
import datetime

from google.cloud import texttospeech as tts

# Info for supported voices: https://cloud.google.com/text-to-speech/docs/voices
VOICE_INFO = {
  "en-US": {
     "f" : {
        "Standard": ["C", "E"],
        "Wavenet": ["C", "E", "F"]
     },
     "m" : {
        "Standard": ["B", "D"],
        "Wavenet": ["A", "B", "D"]
     }
  },
  'en-GB': {
      'f': {
          'Standard': ['A', 'C'],
          'Wavenet': ['A', 'C'],
      },
      'm': {
          'Standard': ['B', 'D'],
          'Wavenet': ['B', 'D'],
      }
  }
}

def generate_voice_name(lang, gender, voice_type):
    return lang + "-" + voice_type + "-" + VOICE_INFO[lang][gender][voice_type][0]

def generate_ssml(text, sentiment):
    """
    Generate an SSML XML based for a specific text and sentiment
    Generally, happier will be higher pitched and faster and more negative is lower pitched.
    Reference: https://www.w3.org/TR/speech-synthesis11/#edef_prosody
    """
    if sentiment == 2:
        # Very positive
        attributes = 'rate="120%" volume="loud" pitch="+3st"'
    elif sentiment == 1:
        # neutral
        attributes = ''
    elif sentiment == 0:
        # very negative
        attributes = 'rate="120%" volume="x-loud" pitch="-6st"'
    else:
        print('Error. invalid sentiment', sentiment)

    # Other possibilities
    # attributes = 'rate="110%" pitch="+2st"'
    # attributes = 'rate="120%" pitch="-3st"'
    return f'<speak><prosody {attributes}>{text}</prosody></speak>'

def synthesize_speech(text_list, sentiment_list, lang='en-GB', gender='f', voice_type='Wavenet', outputPath='output'):
    """
    text: str           The text to be spoken.
    sentiment: float    A float between 0 and 1 indicating how negative or positive the text is.
    """
    client = tts.TextToSpeechClient()

    audio_data += b''
    for i in range(len(text_list)):
        ssml = generate_ssml(text_list[i], sentiment_list[i])
        input_text = tts.types.SynthesisInput(ssml=ssml)

        # Note: the voice can also be specified by name.
        # Names of voices can be retrieved with client.list_voices().
        # for voice in client.list_voices().voices:
        #     if voice.language_codes[0].startswith('en'):
        #         print(voice)
    
        voice_name = generate_voice_name(lang, gender, voice_type)
        voice = tts.types.VoiceSelectionParams(language_code=lang, name=voice_name)

        audio_config = tts.types.AudioConfig(audio_encoding=tts.enums.AudioEncoding.MP3)
        # Synthesize speech and write to output file.
        response = client.synthesize_speech(input_text, voice, audio_config)
        audio_data += response.audio_content

    # The response's audio_content is binary.
    with open(f'{outputPath}.mp3', 'wb') as mp3:
        mp3.write(audio_data)
    with open(f'{outputPath}.txt', 'w') as metadata:
        s = f'Text: {str(text_list)}\nSentiment: {str(sentiment_list)}\nSSML: {ssml}\nVoice name: {voice_name}'
        metadata.write(s)
    print(f'Data saved to {outputPath}')

if __name__ == '__main__':
    # export GOOGLE_APPLICATION_CREDENTIALS=~/.google_cloud_auth.json
    now = str(datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
    for i, sentiment in enumerate([0, 1, 2]):
        synthesize_speech("Don't be angry, uncle. Come! Dine with us to-morrow.", sentiment, outputPath=str(i))
