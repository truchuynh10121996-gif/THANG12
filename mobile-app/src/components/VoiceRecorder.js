import React, { useState } from 'react';
import { TouchableOpacity, StyleSheet, Alert, ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Audio } from 'expo-av';
import { transcribeAudio } from '../services/api';

export default function VoiceRecorder({ language, onTranscriptionComplete }) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [recording, setRecording] = useState(null);

  const startRecording = async () => {
    try {
      // Request permissions
      const { status } = await Audio.requestPermissionsAsync();

      if (status !== 'granted') {
        Alert.alert(
          'Quyền truy cập',
          'Vui lòng cấp quyền truy cập microphone để sử dụng tính năng ghi âm.'
        );
        return;
      }

      // Configure audio mode
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true
      });

      // Start recording
      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );

      setRecording(recording);
      setIsRecording(true);

    } catch (error) {
      console.error('Failed to start recording:', error);
      Alert.alert('Lỗi', 'Không thể bắt đầu ghi âm. Vui lòng thử lại.');
    }
  };

  const stopRecording = async () => {
    try {
      if (!recording) return;

      setIsRecording(false);
      setIsProcessing(true);

      // Stop recording
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();

      // Get audio file
      const audioFile = await fetch(uri);
      const audioBlob = await audioFile.blob();

      // Convert to FormData
      const formData = new FormData();
      formData.append('audio', {
        uri: uri,
        type: 'audio/mp3',
        name: 'recording.mp3'
      });
      formData.append('language', language);

      // Send to API
      const result = await transcribeAudio(formData);

      if (result.transcription && result.transcription.text) {
        onTranscriptionComplete(result.transcription.text);
      } else {
        Alert.alert(
          'Thông báo',
          'Không thể nhận diện giọng nói. Vui lòng thử lại.'
        );
      }

    } catch (error) {
      console.error('Failed to stop recording:', error);
      Alert.alert('Lỗi', 'Không thể xử lý ghi âm. Vui lòng thử lại.');
    } finally {
      setIsProcessing(false);
      setRecording(null);
    }
  };

  const handlePress = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <TouchableOpacity
      style={[
        styles.button,
        isRecording && styles.buttonRecording
      ]}
      onPress={handlePress}
      disabled={isProcessing}
    >
      {isProcessing ? (
        <ActivityIndicator size="small" color="#FF8DAD" />
      ) : (
        <Ionicons
          name={isRecording ? 'stop-circle' : 'mic'}
          size={24}
          color={isRecording ? '#D32F2F' : '#FF8DAD'}
        />
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    padding: 5,
    marginLeft: 5
  },
  buttonRecording: {
    backgroundColor: 'rgba(211, 47, 47, 0.1)',
    borderRadius: 20,
    padding: 8
  }
});
