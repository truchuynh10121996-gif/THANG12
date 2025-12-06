import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  ActivityIndicator,
  Alert
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import ChatBubble from '../components/ChatBubble';
import VoiceRecorder from '../components/VoiceRecorder';
import { sendMessage, synthesizeSpeech } from '../services/api';

export default function ChatScreen({ route, navigation }) {
  const { language = 'vi' } = route.params || {};

  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [conversationId, setConversationId] = useState(null);

  const scrollViewRef = useRef();

  useEffect(() => {
    // Tin nhắn chào mừng
    const welcomeMessage = getWelcomeMessage(language);
    setMessages([
      {
        id: '1',
        text: welcomeMessage,
        sender: 'bot',
        timestamp: new Date()
      }
    ]);
  }, []);

  const getWelcomeMessage = (lang) => {
    const messages = {
      vi: `Xin chào! 👋\n\nTôi là Agribank Digital Guard, trợ lý AI chuyên về phòng chống lừa đảo.\n\nNếu bạn gặp bất kỳ tình huống đáng ngờ nào, hãy mô tả cho tôi. Tôi sẽ phân tích và đưa ra cảnh báo cùng hướng dẫn xử lý an toàn.\n\nBạn có thể nhập văn bản hoặc dùng nút 🎤 để ghi âm giọng nói.`,
      en: `Hello! 👋\n\nI am Agribank Digital Guard, an AI assistant specializing in fraud prevention.\n\nIf you encounter any suspicious situations, please describe them to me. I will analyze and provide warnings along with safe handling instructions.\n\nYou can type text or use the 🎤 button to record your voice.`,
      km: `សួស្តី! 👋\n\nខ្ញុំគឺ Agribank Digital Guard ជំនួយការ AI ឯកទេសខាងការពារការលួចបន្លំ។\n\nប្រសិនបើអ្នកជួបស្ថានភាពសង្ស័យ សូមពណ៌នាដល់ខ្ញុំ។ ខ្ញុំនឹងវិភាគ និងផ្តល់ការព្រមាន រួមជាមួយការណែនាំសុវត្ថិភាព។\n\nអ្នកអាចវាយអក្សរ ឬប្រើប៊ូតុង 🎤 ដើម្បីថតសំឡេង។`
    };
    return messages[lang] || messages.vi;
  };

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage = {
      id: Date.now().toString(),
      text: inputText,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const response = await sendMessage({
        message: inputText,
        conversationId,
        language
      });

      if (!conversationId) {
        setConversationId(response.conversationId);
      }

      const botMessage = {
        id: (Date.now() + 1).toString(),
        text: response.response,
        sender: 'bot',
        timestamp: new Date(),
        isFraudAlert: response.isFraudAlert,
        audioData: null // Will be loaded on demand
      };

      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error('Send message error:', error);
      Alert.alert('Lỗi', 'Không thể gửi tin nhắn. Vui lòng thử lại.');

      const errorMessage = {
        id: (Date.now() + 1).toString(),
        text: 'Xin lỗi, tôi gặp sự cố kết nối. Vui lòng thử lại sau.',
        sender: 'bot',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleVoiceRecorded = (transcription) => {
    if (transcription) {
      setInputText(transcription);
    }
  };

  const scrollToBottom = () => {
    setTimeout(() => {
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 100);
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <LinearGradient
      colors={['#FBD6E3', '#A9EDE9']}
      style={styles.container}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
    >
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity
            onPress={() => navigation.goBack()}
            style={styles.backButton}
          >
            <Ionicons name="arrow-back" size={24} color="#FF8DAD" />
          </TouchableOpacity>

          <View style={styles.headerTitleContainer}>
            <Text style={styles.headerTitle}>Digital Guard</Text>
            <Text style={styles.headerSubtitle}>🛡️ AI Assistant</Text>
          </View>

          <TouchableOpacity style={styles.infoButton}>
            <Ionicons name="information-circle" size={24} color="#FF8DAD" />
          </TouchableOpacity>
        </View>

        {/* Messages */}
        <ScrollView
          ref={scrollViewRef}
          style={styles.messagesContainer}
          contentContainerStyle={styles.messagesContent}
          showsVerticalScrollIndicator={false}
        >
          {messages.map((message) => (
            <ChatBubble
              key={message.id}
              message={message}
              language={language}
            />
          ))}

          {isLoading && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="small" color="#FF8DAD" />
              <Text style={styles.loadingText}>Đang suy nghĩ...</Text>
            </View>
          )}
        </ScrollView>

        {/* Input */}
        <View style={styles.inputContainer}>
          <View style={styles.inputWrapper}>
            <TextInput
              style={styles.input}
              placeholder={
                language === 'vi' ? 'Nhập tin nhắn...' :
                language === 'en' ? 'Type a message...' :
                'វាយសារ...'
              }
              value={inputText}
              onChangeText={setInputText}
              multiline
              maxLength={500}
            />

            <VoiceRecorder
              language={language}
              onTranscriptionComplete={handleVoiceRecorded}
            />

            <TouchableOpacity
              style={[
                styles.sendButton,
                !inputText.trim() && styles.sendButtonDisabled
              ]}
              onPress={handleSendMessage}
              disabled={!inputText.trim() || isLoading}
            >
              <Ionicons
                name="send"
                size={24}
                color={inputText.trim() ? '#FF8DAD' : '#CCC'}
              />
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: 50,
    paddingBottom: 15,
    paddingHorizontal: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.5)'
  },
  backButton: {
    padding: 5
  },
  headerTitleContainer: {
    flex: 1,
    alignItems: 'center'
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FF8DAD'
  },
  headerSubtitle: {
    fontSize: 12,
    color: '#FF6B99'
  },
  infoButton: {
    padding: 5
  },
  messagesContainer: {
    flex: 1
  },
  messagesContent: {
    padding: 15,
    paddingBottom: 10
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    backgroundColor: 'rgba(255, 255, 255, 0.7)',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 10,
    marginTop: 10
  },
  loadingText: {
    marginLeft: 10,
    color: '#FF8DAD',
    fontSize: 14
  },
  inputContainer: {
    paddingHorizontal: 15,
    paddingVertical: 10,
    backgroundColor: 'rgba(255, 255, 255, 0.5)'
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    backgroundColor: '#FFF',
    borderRadius: 25,
    paddingHorizontal: 15,
    paddingVertical: 8,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5
  },
  input: {
    flex: 1,
    maxHeight: 100,
    fontSize: 16,
    color: '#333',
    paddingVertical: 5
  },
  sendButton: {
    marginLeft: 10,
    padding: 5
  },
  sendButtonDisabled: {
    opacity: 0.5
  }
});
