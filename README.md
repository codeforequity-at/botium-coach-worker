# Botium Coach Worker

[![NPM](https://nodei.co/npm/botium-coach-worker.png?downloads=true&downloadRank=true&stars=true)](https://nodei.co/npm/botium-coach-worker/)

[![Codeship Status for codeforequity-at/botium-coach-worker](https://app.codeship.com/projects/9c04d950-c431-0138-30ca-0ef8d48d2ada/status?branch=master)](https://app.codeship.com/projects/406406)
[![npm version](https://badge.fury.io/js/botium-coach-worker.svg)](https://badge.fury.io/js/botium-coach-worker)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)]()

This is a [Botium](https://github.com/codeforequity-at/botium-core) module for extracing and analyzing NLP information from Botium tests.

__Did you read the [Botium in a Nutshell](https://medium.com/@floriantreml/botium-in-a-nutshell-part-1-overview-f8d0ceaf8fb4) articles ? Be warned, without prior knowledge of Botium you won't be able to properly use this library!__

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

