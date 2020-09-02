import os
import logging
from connexion.exceptions import OAuthProblem

validApiTokens = None
if 'AUTH_APIKEYS' in os.environ:
  validApiTokens = os.environ['AUTH_APIKEYS'].split(',')

def apikey_auth(token, required_scopes):
  if validApiTokens is not None:
    if not token in validApiTokens:
      raise OAuthProblem('Invalid token')
  return { 'token': token }
