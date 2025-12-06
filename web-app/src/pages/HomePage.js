import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Button,
  Paper,
  Grid,
  Card,
  CardContent
} from '@mui/material';
import ShieldIcon from '@mui/icons-material/Shield';
import ChatIcon from '@mui/icons-material/Chat';
import SecurityIcon from '@mui/icons-material/Security';
import VerifiedUserIcon from '@mui/icons-material/VerifiedUser';
import TranslateIcon from '@mui/icons-material/Translate';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';

const translations = {
  vi: {
    title: 'Chào mừng đến với Agribank Digital Guard',
    subtitle: 'Trợ lý AI thông minh - Bảo vệ bạn khỏi lừa đảo',
    description: 'Tôi là trợ lý ảo được phát triển bởi Agribank, sẵn sàng giúp bạn nhận diện và phòng tránh các hình thức lừa đảo trực tuyến.',
    startChat: 'Bắt đầu trò chuyện',
    changeLanguage: 'Đổi ngôn ngữ',
    features: {
      title: 'Tính năng nổi bật',
      fraud: {
        title: 'Phát hiện lừa đảo',
        desc: 'Nhận diện các dấu hiệu lừa đảo qua tin nhắn, email và cuộc gọi'
      },
      support: {
        title: 'Hỗ trợ 24/7',
        desc: 'Trợ lý AI luôn sẵn sàng giải đáp mọi thắc mắc của bạn'
      },
      multilang: {
        title: 'Đa ngôn ngữ',
        desc: 'Hỗ trợ Tiếng Việt, English và ភាសាខ្មែរ'
      },
      secure: {
        title: 'Bảo mật cao',
        desc: 'Mọi thông tin trao đổi được mã hóa và bảo mật tuyệt đối'
      }
    }
  },
  en: {
    title: 'Welcome to Agribank Digital Guard',
    subtitle: 'Smart AI Assistant - Protect you from fraud',
    description: 'I am a virtual assistant developed by Agribank, ready to help you identify and prevent online fraud.',
    startChat: 'Start chatting',
    changeLanguage: 'Change language',
    features: {
      title: 'Key Features',
      fraud: {
        title: 'Fraud Detection',
        desc: 'Identify signs of fraud through messages, emails and calls'
      },
      support: {
        title: '24/7 Support',
        desc: 'AI assistant is always ready to answer your questions'
      },
      multilang: {
        title: 'Multilingual',
        desc: 'Support Vietnamese, English and Khmer'
      },
      secure: {
        title: 'High Security',
        desc: 'All exchanged information is encrypted and absolutely secure'
      }
    }
  },
  km: {
    title: 'សូមស្វាគមន៍មកកាន់ Agribank Digital Guard',
    subtitle: 'ជំនួយការ AI ឆ្លាតវៃ - ការពារអ្នកពីការក្លែងបន្លំ',
    description: 'ខ្ញុំជាជំនួយការនិម្មិតដែលបង្កើតឡើងដោយ Agribank រួចរាល់ជួយអ្នកកំណត់អត្តសញ្ញាណនិងការពារការក្លែងបន្លំតាមអ៊ីនធឺណិត។',
    startChat: 'ចាប់ផ្តើមការជជែក',
    changeLanguage: 'ប្តូរភាសា',
    features: {
      title: 'លក្ខណៈពិសេស',
      fraud: {
        title: 'រកឃើញការក្លែងបន្លំ',
        desc: 'កំណត់អត្តសញ្ញាណសញ្ញានៃការក្លែងបន្លំតាមរយៈសារអ៊ីមែលនិងការហៅ'
      },
      support: {
        title: 'ការគាំទ្រ 24/7',
        desc: 'ជំនួយការ AI តែងតែរួចរាល់ឆ្លើយសំណួររបស់អ្នក'
      },
      multilang: {
        title: 'ពហុភាសា',
        desc: 'គាំទ្រភាសាវៀតណាមអង់គ្លេសនិងខ្មែរ'
      },
      secure: {
        title: 'សន្តិសុខខ្ពស់',
        desc: 'ព័ត៌មានដែលបានផ្លាស់ប្តូរទាំងអស់ត្រូវបានអ៊ិនគ្រីបនិងសុវត្ថិភាពដាច់ខាត'
      }
    }
  }
};

function HomePage() {
  const navigate = useNavigate();
  const [language, setLanguage] = useState('vi');

  useEffect(() => {
    const savedLanguage = localStorage.getItem('selectedLanguage');
    if (savedLanguage) {
      setLanguage(savedLanguage);
    } else {
      navigate('/');
    }
  }, [navigate]);

  const t = translations[language] || translations.vi;

  const handleStartChat = () => {
    navigate('/chat');
  };

  const handleChangeLanguage = () => {
    navigate('/');
  };

  const features = [
    {
      icon: <SecurityIcon sx={{ fontSize: 50, color: '#FF8DAD' }} />,
      title: t.features.fraud.title,
      desc: t.features.fraud.desc
    },
    {
      icon: <ChatIcon sx={{ fontSize: 50, color: '#FF8DAD' }} />,
      title: t.features.support.title,
      desc: t.features.support.desc
    },
    {
      icon: <TranslateIcon sx={{ fontSize: 50, color: '#FF8DAD' }} />,
      title: t.features.multilang.title,
      desc: t.features.multilang.desc
    },
    {
      icon: <VerifiedUserIcon sx={{ fontSize: 50, color: '#FF8DAD' }} />,
      title: t.features.secure.title,
      desc: t.features.secure.desc
    }
  ];

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #FFE6F0 0%, #FFC9DD 100%)',
        padding: { xs: 2, md: 4 }
      }}
    >
      <Container maxWidth="lg">
        {/* Hero Section */}
        <Paper
          elevation={6}
          sx={{
            padding: { xs: 3, md: 6 },
            borderRadius: 4,
            background: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(10px)',
            textAlign: 'center',
            mb: 4
          }}
        >
          <ShieldIcon
            sx={{
              fontSize: 100,
              color: '#FF8DAD',
              mb: 2
            }}
          />
          <Typography
            variant="h2"
            sx={{
              fontWeight: 700,
              color: '#FF6B99',
              mb: 2,
              fontSize: { xs: '2rem', md: '3rem' }
            }}
          >
            {t.title}
          </Typography>
          <Typography
            variant="h5"
            sx={{
              color: '#666',
              mb: 3,
              fontWeight: 500,
              fontSize: { xs: '1.2rem', md: '1.5rem' }
            }}
          >
            {t.subtitle}
          </Typography>
          <Typography
            variant="body1"
            sx={{
              color: '#888',
              mb: 4,
              maxWidth: 600,
              margin: '0 auto',
              mb: 4
            }}
          >
            {t.description}
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
            <Button
              variant="contained"
              size="large"
              endIcon={<ArrowForwardIcon />}
              onClick={handleStartChat}
              sx={{
                bgcolor: '#FF8DAD',
                color: 'white',
                padding: '12px 32px',
                borderRadius: 3,
                fontSize: '1.1rem',
                fontWeight: 600,
                boxShadow: '0 4px 15px rgba(255, 141, 173, 0.4)',
                '&:hover': {
                  bgcolor: '#FF6B99',
                  transform: 'translateY(-2px)',
                  boxShadow: '0 6px 20px rgba(255, 141, 173, 0.5)'
                }
              }}
            >
              {t.startChat}
            </Button>
            <Button
              variant="outlined"
              size="large"
              startIcon={<TranslateIcon />}
              onClick={handleChangeLanguage}
              sx={{
                borderColor: '#FF8DAD',
                color: '#FF6B99',
                padding: '12px 32px',
                borderRadius: 3,
                fontSize: '1.1rem',
                fontWeight: 600,
                borderWidth: 2,
                '&:hover': {
                  borderWidth: 2,
                  borderColor: '#FF6B99',
                  backgroundColor: '#FFE6F0'
                }
              }}
            >
              {t.changeLanguage}
            </Button>
          </Box>
        </Paper>

        {/* Features Section */}
        <Typography
          variant="h4"
          sx={{
            fontWeight: 700,
            color: '#FF6B99',
            mb: 3,
            textAlign: 'center'
          }}
        >
          {t.features.title}
        </Typography>
        <Grid container spacing={3}>
          {features.map((feature, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card
                sx={{
                  height: '100%',
                  borderRadius: 3,
                  background: 'rgba(255, 255, 255, 0.95)',
                  backdropFilter: 'blur(10px)',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-8px)',
                    boxShadow: '0 8px 25px rgba(255, 141, 173, 0.3)'
                  }
                }}
              >
                <CardContent sx={{ textAlign: 'center', padding: 3 }}>
                  <Box sx={{ mb: 2 }}>{feature.icon}</Box>
                  <Typography
                    variant="h6"
                    sx={{
                      fontWeight: 600,
                      color: '#FF6B99',
                      mb: 1
                    }}
                  >
                    {feature.title}
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#888' }}>
                    {feature.desc}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
}

export default HomePage;
