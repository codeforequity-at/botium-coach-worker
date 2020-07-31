# Nlp Analytics information from Chatbot Convesations.

[![NPM](https://nodei.co/npm/botium-cli.png?downloads=true&downloadRank=true&stars=true)](https://nodei.co/npm/botium-cli/)

[ ![Codeship Status for codeforequity-at/botium-cli](https://app.codeship.com/projects/4d7fd410-18ab-0136-6ab1-6e2b4bb62b94/status?branch=master)](https://app.codeship.com/projects/283938)
[![npm version](https://badge.fury.io/js/botium-cli.svg)](https://badge.fury.io/js/botium-cli)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)]()

## Installation

```
> npm install botium-coach-worker -save
```

## Usage

```
const nlpAnalyticsData = nlpAnalytics.process({testCaseResults, connectorFeatures})
```

### Input

#### testCaseResults

List of conversations. __Botium Coach Worker__ supports now just the test results output format of __Botium Box__.

See some examples in _test/dynamic_ directory with _*.input.json_ postfix

#### connectorFeatures

The features of the chatbot. If it is not set, then __Botium Coach Worker__ tries to calculate them, but it is in some cases not possible.
In this case the *Supported flags like _intentConfidenceSupported_ in _nlpAnalyticsData.overallStat_ may be null.

(See some examples in _test/dynamic_ directory with _*.connectorFeatures.json_ postfix)

### Output

Nlp Analytics data. __Botium Coach Worker__ supports now just the Nlp Analytics input format of __Botium Box__.

