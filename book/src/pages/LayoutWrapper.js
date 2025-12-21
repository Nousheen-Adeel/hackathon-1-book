import React from 'react';
import Layout from '@theme/Layout';
import ChatbotIntegration from '@site/src/components/ChatbotIntegration';

export default function LayoutWrapper(props) {
  return (
    <Layout {...props}>
      {props.children}
      <ChatbotIntegration />
    </Layout>
  );
}