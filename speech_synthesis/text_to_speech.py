"""Synthesizes speech from the input string of text or ssml.

Note: ssml must be well-formed according to:
    https://www.w3.org/TR/speech-synthesis/
"""
import datetime

from google.cloud import texttospeech as tts

##Hardcoded list for supported list"
'''
VOICE={
  "en-US": {
     "f" : {
        "Standard": ["C", "E"],
        "Wavenet": ["C", "E", "F"]
     },
     "m" : {
        "Standard": ["B", "D"],
        "Wavenet": ["A", "B", "D"]
     }
  }
}

def clip_value(val, lo, hi):
    return min(hi, max(val, lo))

def voiceOutput(pitchInput, genderInput, textInput):#, filename):
    #with open(filename, 'wb') as out:
        # Set the text input to be synthesized
        audio_data = b''
        for i in range (len(textInput)):
            # Instantiates a client
            client = texttospeech.TextToSpeechClient()
            lang = 'en' #detectLanguage(textInput[i])
            lang = matchCountryCode(lang)

            synthesis_input = texttospeech.types.SynthesisInput(text=textInput[i])

            if (genderInput[i] == 'f'):
                gender = texttospeech.enums.SsmlVoiceGender.FEMALE
            else:
                gender = texttospeech.enums.SsmlVoiceGender.MALE

            name_input=returnVoiceName(lang, genderInput[i], "Wavenet")

            print (name_input);
            # Build the voice request, select the language code ("en-US") and the ssml
            # voice gender ("neutral")
            voice = texttospeech.types.VoiceSelectionParams(
                language_code=lang,
                name=name_input,
                ssml_gender=gender)

            pace = 0.8
            if abs(pitchInput[i]) >= 6:
                pace += 0.2

            # Select the type of audio file you want returned
            audio_config = texttospeech.types.AudioConfig(
                audio_encoding=texttospeech.enums.AudioEncoding.MP3,
                pitch=clip_value(pitchInput[i] - 3, -20, 20),
                speaking_rate=pace) # 1 is the normal speed

            # Perform the text-to-speech request on the text input with the selected
            # voice parameters and audio file type
            response = client.synthesize_speech(synthesis_input, voice, audio_config)

            #responseArray.extend(response)

            #out.write(response.audio_content)
            audio_data += response.audio_content
        return audio_data
        # The response's audio_content is binary.
            # Write the response to the output file.
            #for response in responseArray:

    #print("Audio content written to file " + filename)


def matchCountryCode(lang):
    if lang in ["nl", "en", "fr", "de", "it", "ja", "ko", "es", "pt", "sv", "tr"]:
        ##Some default accents
        if (lang == "en"):
            return "en-US"
        elif (lang == "ja"):
            return "ja-JP"
        elif (lang == "ko"):
            return "ko-KR"
        elif (lang=="sv"):
            return "sv-SE"
        else:
            return lang+"-"+lang.upper()
    else: ##Not supported by google
        return "en-US"

def returnVoiceName(lang, gender, voiceType):
    ##Currently by default return the first one available.
    return lang + "-" + voiceType + "-" + VOICE[lang][gender][voiceType][0]

'''

def synthesize_speech(text, sentiment, gender='F'):
    """
    text: str           The text to be spoken.
    sentiment: float    A float between 0 and 1 indicating how negative or positive the text is.
    """
    now = str(datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
    client = tts.TextToSpeechClient()

    input_text = tts.types.SynthesisInput(text=text)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    # print(client.list_voices())
    voice = tts.types.VoiceSelectionParams(
        language_code='en-US',
        ssml_gender=tts.enums.SsmlVoiceGender.FEMALE if gender == 'F' else tts.enums.SsmlVoiceGender.Male)

    audio_config = tts.types.AudioConfig(audio_encoding=tts.enums.AudioEncoding.MP3)
    # Synthesize speech and write to output file.
    response = client.synthesize_speech(input_text, voice, audio_config)
    # The response's audio_content is binary.
    with open(f'outputs/{now}.mp3', 'wb') as mp3:
        mp3.write(response.audio_content)
    with open(f'outputs/{now}.meta', 'w') as metadata:
        metadata.writelines(['Text: ', text, '\nSentiment: ', str(sentiment)])
    print(f'Data saved to outputs/{now}')

if __name__ == '__main__':
    # export GOOGLE_APPLICATION_CREDENTIALS=~/.google_cloud_auth.json
    synthesize_speech('hello world', 0.5)
