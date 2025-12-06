import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Container, Typography, Button, Paper, Grid } from '@mui/material';
import LanguageIcon from '@mui/icons-material/Language';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';

const languages = [
  {
    code: 'vi',
    name: 'Tiếng Việt',
    flag: '🇻🇳',
    greeting: 'Xin chào!'
  },
  {
    code: 'en',
    name: 'English',
    flag: '🇬🇧',
    greeting: 'Hello!'
  },
  {
    code: 'km',
    name: 'ភាសាខ្មែរ',
    flag: '🇰🇭',
    greeting: 'សួស្តី!'
  }
];

function LanguagePage() {
  const navigate = useNavigate();

  const handleLanguageSelect = (languageCode) => {
    localStorage.setItem('selectedLanguage', languageCode);
    navigate('/home');
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #FFE6F0 0%, #FFC9DD 100%)',
        padding: 2
      }}
    >
      <Container maxWidth="md">
        <Paper
          elevation={6}
          sx={{
            padding: { xs: 3, md: 5 },
            borderRadius: 4,
            background: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(10px)'
          }}
        >
          <Box sx={{ textAlign: 'center', mb: 4 }}>
            <LanguageIcon
              sx={{
                fontSize: 80,
                color: '#FF8DAD',
                mb: 2
              }}
            />
            <Typography
              variant="h3"
              sx={{
                fontWeight: 700,
                color: '#FF6B99',
                mb: 1
              }}
            >
              Agribank Digital Guard
            </Typography>
            <Typography
              variant="h6"
              sx={{
                color: '#666',
                fontWeight: 400
              }}
            >
              Chọn ngôn ngữ của bạn / Choose your language
            </Typography>
          </Box>

          <Grid container spacing={3}>
            {languages.map((lang) => (
              <Grid item xs={12} md={4} key={lang.code}>
                <Button
                  fullWidth
                  variant="outlined"
                  onClick={() => handleLanguageSelect(lang.code)}
                  sx={{
                    padding: 3,
                    borderRadius: 3,
                    borderWidth: 2,
                    borderColor: '#FF8DAD',
                    color: '#FF6B99',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      borderWidth: 2,
                      borderColor: '#FF6B99',
                      backgroundColor: '#FFE6F0',
                      transform: 'translateY(-4px)',
                      boxShadow: '0 8px 20px rgba(255, 141, 173, 0.3)'
                    }
                  }}
                >
                  <Box>
                    <Typography variant="h2" sx={{ mb: 1 }}>
                      {lang.flag}
                    </Typography>
                    <Typography
                      variant="h6"
                      sx={{ fontWeight: 600, mb: 0.5 }}
                    >
                      {lang.name}
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#888' }}>
                      {lang.greeting}
                    </Typography>
                  </Box>
                </Button>
              </Grid>
            ))}
          </Grid>

          <Box sx={{ textAlign: 'center', mt: 4 }}>
            <Typography variant="body2" sx={{ color: '#999' }}>
              Powered by Google Generative AI
            </Typography>
          </Box>
        </Paper>
      </Container>
    </Box>
  );
}

export default LanguagePage;
