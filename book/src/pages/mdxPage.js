import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import { ChatbotIntegration } from '../components/ChatbotIntegration';

export default function MDXPage(props) {
  const { content: MDXContent } = props;
  const {
    metadata: { title, description },
  } = MDXContent;

  return (
    <Layout title={title} description={description}>
      <main className="container margin-vert--lg">
        <div className="row">
          <div className={clsx('col', 'col--8', 'col--offset-2')}>
            <MDXContent />
          </div>
        </div>
      </main>
      <ChatbotIntegration />
    </Layout>
  );
}