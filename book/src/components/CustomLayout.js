import React from 'react';
import Layout from '@theme/Layout';
import Chatbot from '../components/Chatbot';

// Custom layout wrapper that includes the chatbot
export default function CustomLayout(props) {
  return (
    <>
      <Layout {...props}>
        {props.children}
      </Layout>
      <Chatbot />
    </>
  );
}