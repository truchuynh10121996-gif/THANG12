import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { Audio } from 'expo-av';
import { synthesizeSpeech } from '../services/api';

export default function ChatBubble({ message, language }) {
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);
  const [sound, setSound] = useState(null);

  const isBot = message.sender === 'bot';
  const isFraud = message.isFraudAlert;

  const handlePlayAudio = async () => {
    try {
      if (isPlayingAudio && sound) {
        // Stop audio
        await sound.stopAsync();
        await sound.unloadAsync();
        setSound(null);
        setIsPlayingAudio(false);
        return;
      }

      setIsLoadingAudio(true);

      // Get audio from API
      const audioData = await synthesizeSpeech({
        text: message.text,
        language,
        gender: 'FEMALE'
      });

      // Convert base64 to audio
      const { sound: audioSound } = await Audio.Sound.createAsync(
        { uri: `data:audio/mp3;base64,${audioData.audioContent}` },
        { shouldPlay: true }
      );

      setSound(audioSound);
      setIsPlayingAudio(true);

      // Handle playback status
      audioSound.setOnPlaybackStatusUpdate((status) => {
        if (status.didJustFinish) {
          setIsPlayingAudio(false);
          audioSound.unloadAsync();
          setSound(null);
        }
      });

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
  }
});
