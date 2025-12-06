import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Platform,
  Image
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { Audio } from 'expo-av';
import * as Speech from 'expo-speech';
import { synthesizeSpeech } from '../services/api';

export default function ChatBubble({ message, language }) {
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);
  const [sound, setSound] = useState(null);

  const isBot = message.sender === 'bot';
  const isFraud = message.isFraudAlert;

  // Map ngôn ngữ sang mã BCP-47 cho Speech
  const getLanguageCode = (lang) => {
    const languageMap = {
      vi: 'vi-VN',
      en: 'en-US',
      km: 'km-KH'
    };
    return languageMap[lang] || 'vi-VN';
  };

  // Sử dụng expo-speech (native TTS) làm fallback
  const speakWithNativeTTS = async (text) => {
    const langCode = getLanguageCode(language);

    return new Promise((resolve, reject) => {
      Speech.speak(text, {
        language: langCode,
        pitch: 1.0,
        rate: 0.9,
        onStart: () => {
          setIsPlayingAudio(true);
        },
        onDone: () => {
          setIsPlayingAudio(false);
          resolve();
        },
        onStopped: () => {
          setIsPlayingAudio(false);
          resolve();
        },
        onError: (error) => {
          setIsPlayingAudio(false);
          reject(error);
        }
      });
    });
  };

  const handlePlayAudio = async () => {
    try {
      // Nếu đang phát, dừng lại
      if (isPlayingAudio) {
        if (sound) {
          await sound.stopAsync();
          await sound.unloadAsync();
          setSound(null);
        }
        // Dừng expo-speech nếu đang phát
        Speech.stop();
        setIsPlayingAudio(false);
        return;
      }

      setIsLoadingAudio(true);

      // Thử lấy audio từ API backend trước
      try {
        const audioData = await synthesizeSpeech({
          text: message.text,
          language,
          gender: 'FEMALE'
        });

        // Kiểm tra xem response có phải là audio hợp lệ không
        if (audioData && audioData.audioContent) {
          // Thử decode base64 để kiểm tra có phải audio hay không
          // Nếu response là JSON error thì base64 sẽ chứa JSON, không phải audio MP3
          const base64Str = audioData.audioContent;

          // Kiểm tra header của MP3 (ID3 tag hoặc MPEG sync word)
          // MP3 files bắt đầu với "ID3" hoặc 0xFF 0xFB/0xFA/0xF3/0xF2
          const isValidMP3 = base64Str.startsWith('SUQz') || // "ID3" in base64
                            base64Str.startsWith('//') ||    // 0xFF in base64
                            base64Str.startsWith('/+');      // 0xFF variants

          if (isValidMP3) {
            // Audio hợp lệ từ backend
            const { sound: audioSound } = await Audio.Sound.createAsync(
              { uri: `data:audio/mp3;base64,${base64Str}` },
              { shouldPlay: true }
            );

            setSound(audioSound);
            setIsPlayingAudio(true);

            audioSound.setOnPlaybackStatusUpdate((status) => {
              if (status.didJustFinish) {
                setIsPlayingAudio(false);
                audioSound.unloadAsync();
                setSound(null);
              }
            });

            setIsLoadingAudio(false);
            return;
          }
        }

        // Nếu không hợp lệ, throw error để fallback
        throw new Error('Invalid audio response from backend');

      } catch (backendError) {
        console.log('Backend TTS failed, using native TTS fallback:', backendError.message);
      }

      // Fallback: Sử dụng expo-speech (native TTS)
      await speakWithNativeTTS(message.text);

    } catch (error) {
      console.error('Play audio error:', error);
      setIsPlayingAudio(false);
    } finally {
      setIsLoadingAudio(false);
    }
  };

  const formatTime = (date) => {
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return `${hours}:${minutes}`;
  };

  return (
    <View style={[styles.container, isBot ? styles.botContainer : styles.userContainer]}>
      {isBot ? (
        <View style={styles.botBubbleWrapper}>
          {isFraud && (
            <View style={styles.alertBadge}>
              <Ionicons name="warning" size={16} color="#FFF" />
              <Text style={styles.alertBadgeText}>CẢNH BÁO</Text>
            </View>
          )}

          <View style={[styles.bubble, styles.botBubble, isFraud && styles.fraudBubble]}>
            <Text style={[styles.text, styles.botText]}>{message.text}</Text>

            <View style={styles.bubbleFooter}>
              <Text style={styles.time}>{formatTime(message.timestamp)}</Text>

              <TouchableOpacity
                style={styles.speakerButton}
                onPress={handlePlayAudio}
                disabled={isLoadingAudio}
              >
                {isLoadingAudio ? (
                  <ActivityIndicator size="small" color="#FF8DAD" />
                ) : (
                  <Ionicons
                    name={isPlayingAudio ? 'stop-circle' : 'volume-high'}
                    size={20}
                    color="#FF8DAD"
                  />
                )}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      ) : (
        <LinearGradient
          colors={['#FF8DAD', '#FF6B99']}
          style={[styles.bubble, styles.userBubble]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
        >
          {/* Hiển thị ảnh nếu có */}
          {message.hasImage && message.imageUri && (
            <Image
              source={{ uri: message.imageUri }}
              style={styles.messageImage}
              resizeMode="cover"
            />
          )}
          <Text style={[styles.text, styles.userText]}>{message.text}</Text>
          <Text style={[styles.time, styles.userTime]}>{formatTime(message.timestamp)}</Text>
        </LinearGradient>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginBottom: 15,
    maxWidth: '80%'
  },
  botContainer: {
    alignSelf: 'flex-start'
  },
  userContainer: {
    alignSelf: 'flex-end'
  },
  botBubbleWrapper: {
    position: 'relative'
  },
  alertBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#D32F2F',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 15,
    marginBottom: 5,
    alignSelf: 'flex-start'
  },
  alertBadgeText: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: 'bold',
    marginLeft: 5
  },
  bubble: {
    borderRadius: 20,
    padding: 12,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1
    },
    shadowOpacity: 0.22,
    shadowRadius: 2.22,
    elevation: 3
  },
  botBubble: {
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    borderBottomLeftRadius: 5
  },
  fraudBubble: {
    borderWidth: 2,
    borderColor: '#D32F2F'
  },
  userBubble: {
    borderBottomRightRadius: 5
  },
  text: {
    fontSize: 15,
    lineHeight: 22
  },
  botText: {
    color: '#333'
  },
  userText: {
    color: '#FFF'
  },
  bubbleFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 5
  },
  time: {
    fontSize: 11,
    color: '#999',
    marginTop: 5
  },
  userTime: {
    color: 'rgba(255, 255, 255, 0.7)'
  },
  speakerButton: {
    padding: 5
  },
  messageImage: {
    width: '100%',
    height: 150,
    borderRadius: 10,
    marginBottom: 8
  }
});
